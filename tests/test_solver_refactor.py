"""Regression tests for the solver refactor."""

from __future__ import annotations

import inspect
import random
import shutil
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pytest
import torch

from buildings import LargeCommercialBuilding2

try:
    import cma  # noqa: F401
except ImportError:
    sys.modules.setdefault("cma", types.ModuleType("cma"))

import L2Otraining
import NN_solver
import NN_solver_old
import data_generation
import follower_solver
import follower_solver_old
import trust_region_solver
import trust_region_solver_old
import zero_solver
import zero_solver_old
import stackelberg.trust_region as trust_region_module
from stackelberg.follower import FollowerProblem
from stackelberg.followers import CompositeFollowerProblem
from stackelberg.followers.bg import (
    BUILDING_TYPES,
    BuildingGroupFollowerProblem,
    create_cigre_residential_network,
    extract_bg_constants,
)
from stackelberg.followers.sb import extract_sb_constants
from stackelberg.followers.ip import (
    IndustrialParkFollowerProblem,
    create_industrial_park_network,
)
from stackelberg.network import create_case33bw_20kv
from stackelberg.instances import (
    DEFAULT_INSTANCE_IDS,
    resolve_follower_configuration,
)
from stackelberg.randomness import set_global_seed
from stackelberg.distflow import torch_penalty_distflow
from stackelberg.trust_region import (
    CompositeGlobalSurrogate,
    GlobalSurrogate,
    ResidualModel,
    TRResidualController,
    TrustRegionExperimentConfig,
    build_trust_region_experiment,
    save_trust_region_true_responses,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _assert_dict_equal(old: dict, new: dict) -> None:
    assert old.keys() == new.keys()
    for key, old_value in old.items():
        new_value = new[key]
        if isinstance(old_value, np.ndarray):
            np.testing.assert_allclose(old_value, new_value)
        else:
            assert old_value == new_value


def test_legacy_snapshots_do_not_import_refactored_package() -> None:
    for name in ("NN_solver", "follower_solver", "zero_solver", "trust_region_solver"):
        source = (PROJECT_ROOT / "old code" / f"{name}_old.py").read_text(
            encoding="utf-8"
        )
        assert "from stackelberg" not in source
        assert "import stackelberg" not in source


def test_canonical_modules_do_not_import_legacy_snapshots() -> None:
    for module in (NN_solver, follower_solver, zero_solver, trust_region_solver):
        assert "_old" not in inspect.getsource(module)


def test_profiles_and_follower_constants_match_legacy() -> None:
    lv_net = pn.create_cigre_network_lv()
    for scenario in ("SB", "BG", "IP"):
        np.testing.assert_allclose(
            follower_solver_old._load_curve(24, scenario),
            follower_solver._load_curve(24, scenario),
        )
    _assert_dict_equal(
        follower_solver_old.extract_constants(lv_net, 24, "SB"),
        follower_solver.extract_constants(lv_net, 24, "SB"),
    )
    bg_solver = follower_solver.build_solver("BG", T=24)
    assert bg_solver.constants["n_building"] == 17
    ip_solver = follower_solver.build_solver("IP", T=24)
    assert ip_solver.constants["n_storage"] == ip_solver.constants["n_stage"] - 1
    assert follower_solver.build_solver("IE", T=24).scenario == "IP"
    assert not issubclass(
        follower_solver.BuildingGroupSolver, follower_solver.SingleBusSolver
    )
    assert not issubclass(
        follower_solver.IndustrialParkSolver, follower_solver.SingleBusSolver
    )


def test_named_instance_configuration_is_heterogeneous_and_extensible() -> None:
    labels, scenario_types, follower_kwargs = resolve_follower_configuration(
        DEFAULT_INSTANCE_IDS
    )

    assert labels == ["SB1", "SB2", "BG1", "BG2", "IP1"]
    assert scenario_types == ["SB", "SB", "BG", "BG", "IP"]
    assert follower_kwargs[0] == {}
    assert follower_kwargs[1]["load_scale"] == pytest.approx(1.10)
    assert follower_kwargs[2]["date"] == "2023-01-01"
    assert follower_kwargs[3]["date"] == "2023-07-01"
    assert follower_kwargs[4]["date"] == "2023-01-01"

    _, _, overridden = resolve_follower_configuration(
        ["BG2"], [{"date": "2023-08-01", "storage_bus_ids": [2]}]
    )
    assert overridden == [{"date": "2023-08-01", "storage_bus_ids": [2]}]


def test_sb2_scales_only_load_by_ten_percent() -> None:
    net = pn.create_cigre_network_lv()
    sb1 = extract_sb_constants(net, 24)
    sb2 = extract_sb_constants(net, 24, load_scale=1.10)

    np.testing.assert_allclose(sb2["p_load"], 1.10 * sb1["p_load"])
    np.testing.assert_allclose(sb2["q_load"], 1.10 * sb1["q_load"])
    for key in (
        "p_mg_max",
        "s_max",
        "p_ch_max",
        "p_dis_max",
        "e_max",
        "e_init",
    ):
        assert sb2[key] == pytest.approx(sb1[key])


def test_bg_instances_use_different_days() -> None:
    net = create_cigre_residential_network()
    bg1 = extract_bg_constants(net, 24, date="2023-07-15")
    bg2 = extract_bg_constants(net, 24, date="2023-07-16")

    assert bg1["profile_date"] == "2023-07-15"
    assert bg2["profile_date"] == "2023-07-16"
    assert not np.allclose(bg1["p_load"], bg2["p_load"])
    assert not np.allclose(bg1["p_pv_building"], bg2["p_pv_building"])


def test_sb_solver_accepts_failure_and_metadata_options() -> None:
    solver = follower_solver.build_solver(
        "SB", T=24, failure_mode="none", include_metadata=False
    )

    assert solver.failure_mode == "none"
    assert solver.include_metadata is False


def test_ip_solver_respects_process_inventory_and_power_balances() -> None:
    T = 24
    problem = IndustrialParkFollowerProblem(T=T)
    result = problem.lower_solve(np.full(T, 0.5))
    constants = problem.constants

    assert constants["n_pline"] == 2
    assert "n_line" not in constants
    assert constants["pline_names"] == (
        "Production Line 1",
        "Production Line 2",
    )
    assert constants["n_bus"] == 4
    assert constants["n_branch"] == 3
    assert constants["pcc"] == 0
    assert constants["der_bus"] == 1
    assert constants["pv_bus"] == 1
    assert constants["battery_bus"] == 1
    assert constants["pline_buses"] == [2, 3]
    assert constants["s_base_mva"] == 10.0
    assert constants["v_base_kv"] == 20.0
    np.testing.assert_allclose(
        constants["branch_s_max_mva"], [12.5, 4.5, 9.0]
    )
    assert constants["process_names"] == ("C", "KFP", "CP", "FG")
    np.testing.assert_allclose(
        constants["varphi_max"],
        np.array([[100.0, 25.0, 30.0, 35.0], [200.0, 50.0, 60.0, 70.0]]),
    )
    np.testing.assert_allclose(
        constants["p_tilde"],
        constants["p_process_max_mw"] / constants["varphi_max"],
    )
    np.testing.assert_allclose(
        constants["eta"],
        np.array([[1.5, 1.2, 1.2, 1.2], [1.5, 1.2, 1.2, 1.2]]),
    )
    np.testing.assert_allclose(constants["sos_init"], 0.5 * constants["sos_max"])
    np.testing.assert_array_equal(constants["min_on_time"], 8)
    constraint_names = {
        constraint.name for constraint in problem._model.getConstrs()
    }
    for pline in range(constants["n_pline"]):
        for stage, process_name in enumerate(constants["process_names"]):
            for t in range(T):
                ramp_up_name = f"ramp_up[{pline},{stage},{t}]"
                if process_name == "FG":
                    assert ramp_up_name not in constraint_names
                else:
                    assert ramp_up_name in constraint_names
    raw_pv_profile = LargeCommercialBuilding2("2023-01-01").get_pv_curve()
    np.testing.assert_allclose(
        constants["p_pv_hat"],
        raw_pv_profile / raw_pv_profile.max() * 2.0,
    )
    assert constants["p_pv_peak_mw"] == 2.0

    assert result is not None
    assert result["mip_start_used"] is False
    assert result["p_mg"].shape == (T,)
    assert np.all(result["p_mg"] >= -1e-8)
    assert np.all(result["p_mg"] <= constants["p_mg_max"] + 1e-8)
    np.testing.assert_allclose(
        result["varphi"],
        constants["eta"][:, :, None] * result["phi"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result["production_power"],
        constants["p_tilde"][:, :, None] * result["varphi"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result["sos"][:, :, 0],
        constants["sos_init"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result["sos"][:, :, -1],
        result["sos"][:, :, 0],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result["sos"][:, :, 1:],
        result["sos"][:, :, :-1]
        + result["varphi"][:, :-1, :]
        - result["phi"][:, 1:, :]
        + result["net_transfer"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result["p_mg"] + result["p_pv"] + result["discharge"],
        result["p_prod"] + result["charge"],
        atol=1e-8,
    )
    assert np.all(
        result["varphi"][:, -1, :].sum(axis=1)
        >= constants["q_pline_min"] - 1e-8
    )
    assert result["voltage"].shape == (4, T)
    assert result["p_flow"].shape == (3, T)
    assert result["q_flow"].shape == (3, T)
    np.testing.assert_allclose(
        result["p_flow"][0] * constants["s_base_mva"],
        result["p_mg"],
        atol=1e-8,
    )
    pline_power = result["production_power"].sum(axis=1)
    np.testing.assert_allclose(
        result["p_flow"][1:] * constants["s_base_mva"],
        pline_power,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result["voltage"][constants["pcc"]],
        constants["v_ref"] ** 2,
        atol=1e-8,
    )
    assert np.all(result["voltage"] >= constants["v_min"] - 1e-8)
    assert np.all(result["voltage"] <= constants["v_max"] + 1e-8)
    p_flow_mw = result["p_flow"] * constants["s_base_mva"]
    q_flow_mvar = result["q_flow"] * constants["s_base_mva"]
    assert np.all(p_flow_mw >= -1e-8)
    assert np.all(q_flow_mvar >= -1e-8)
    assert np.all(
        p_flow_mw <= constants["branch_s_max_mva"][:, None] + 1e-8
    )
    assert np.all(
        q_flow_mvar <= constants["branch_s_max_mva"][:, None] + 1e-8
    )

    composite_result = CompositeFollowerProblem([problem], T=T).lower_solve(
        np.full(T, 0.5)
    )
    assert composite_result is not None
    assert composite_result["p_mg"].shape == (1, T)
    assert composite_result["follower_results"][0]["mip_start_used"] is True


def test_industrial_park_network_has_expected_bus_allocation() -> None:
    net = create_industrial_park_network()

    assert net.sn_mva == 10.0
    assert net.bus["name"].tolist() == [
        "PCC 0",
        "Bus 1 - PV and BESS",
        "Bus 2 - Production Line 1",
        "Bus 3 - Production Line 2",
    ]
    assert net.line[["from_bus", "to_bus"]].to_numpy().tolist() == [
        [0, 1],
        [1, 2],
        [1, 3],
    ]


def test_bg_constants_map_and_scale_all_non_pcc_buses() -> None:
    constants = BuildingGroupFollowerProblem().constants

    assert constants["n_bus"] == 18
    assert constants["n_line"] == 17
    assert constants["n_building"] == 17
    assert constants["n_storage"] == 3
    assert constants["n_storage_binary"] == 72
    assert len(set(constants["storage_bus_ids"])) == 3
    assert set(constants["storage_bus_ids"]) <= set(constants["building_bus_ids"])
    assert np.count_nonzero(constants["e_max"]) == 3
    assert len(set(constants["building_buses"])) == 17
    assert constants["pcc"] not in constants["building_buses"]
    assert constants["building_classes"] == [
        BUILDING_TYPES[index % len(BUILDING_TYPES)].__name__
        for index in range(constants["n_building"])
    ]
    assert constants["p_load_building"].shape == (17, 24)
    assert constants["p_pv_building"].shape == (17, 24)
    assert constants["outdoor_temperature"].shape == (17, 24)
    np.testing.assert_allclose(constants["beta_heat"], constants["beta_cool"])
    np.testing.assert_allclose(
        constants["p_load"].mean(), constants["base_p_bus"].sum()
    )


def _two_bus_bg_network() -> pp.pandapowerNet:
    net = pp.create_empty_network(sn_mva=1.0)
    pcc = pp.create_bus(net, vn_kv=0.4, name="Bus R1")
    building_bus = pp.create_bus(net, vn_kv=0.4, name="Bus R2")
    pp.create_ext_grid(net, bus=pcc)
    pp.create_line_from_parameters(
        net,
        pcc,
        building_bus,
        length_km=0.01,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.05,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        name="R1-R2",
    )
    pp.create_load(net, bus=building_bus, p_mw=0.05, q_mvar=0.01)
    return net


def test_bg_solver_respects_device_and_network_bounds() -> None:
    problem = BuildingGroupFollowerProblem(net=_two_bus_bg_network())
    result = problem.lower_solve(np.full(24, 0.5))
    constants = problem.constants

    assert result is not None
    assert result["p_mg"].shape == (24,)
    assert np.all(result["p_mg"] >= -1e-8)
    assert np.all(result["p_mg"] <= constants["p_mg_max"] + 1e-8)
    assert np.all(result["energy"] >= constants["e_min"][:, None] - 1e-8)
    assert np.all(result["energy"] <= constants["e_max"][:, None] + 1e-8)
    assert np.all(
        result["indoor_temperature"] >= constants["tau_min"][:, None] - 1e-8
    )
    assert np.all(
        result["indoor_temperature"] <= constants["tau_max"][:, None] + 1e-8
    )
    np.testing.assert_allclose(
        result["energy"][:, -1], constants["e_init"], atol=1e-8
    )
    np.testing.assert_allclose(
        result["indoor_temperature"][:, -1],
        constants["tau_init"],
        atol=1e-8,
    )
    assert np.all(result["voltage"] >= constants["v_min"] - 1e-8)
    assert np.all(result["voltage"] <= constants["v_max"] + 1e-8)
    assert np.all(result["current"] <= constants["l_max"][:, None] + 1e-8)
    assert np.all(np.minimum(result["p_ch"], result["p_dis"]) <= 1e-8)
    assert np.all(
        result["p_heat"] / constants["p_heat_max"][:, None]
        + result["p_cool"] / constants["p_cool_max"][:, None]
        <= 1.0 + 1e-8
    )
    assert np.all(np.minimum(result["p_heat"], result["p_cool"]) <= 1e-8)


def test_upper_constants_are_reproducible_and_radial() -> None:
    upper_net = pn.case33bw()
    for extract in (
        zero_solver.extract_constants,
        NN_solver.extract_constants,
        trust_region_solver.extract_constants,
    ):
        set_global_seed(123)
        first = extract({"mg_buses": [18]}, upper_net, 24)
        set_global_seed(123)
        second = extract({"mg_buses": [18]}, upper_net, 24)
        np.testing.assert_array_equal(first["lambda_g"], second["lambda_g"])
        assert first["n_line"] == first["n_bus"] - 1
        assert first["mg_buses"] == [first["bus_lookup"][18]]
        assert all(
            0 <= source < first["n_bus"] and 0 <= target < first["n_bus"]
            for source, target in zip(first["from_bus"], first["to_bus"])
        )


def test_case33bw_20kv_preserves_per_unit_line_impedances() -> None:
    original = pn.case33bw()
    converted = create_case33bw_20kv()

    assert np.allclose(converted.bus["vn_kv"], 20.0)
    original_z_base = float(original.bus["vn_kv"].iloc[0]) ** 2 / original.sn_mva
    converted_z_base = 20.0**2 / converted.sn_mva
    for column in ("r_ohm_per_km", "x_ohm_per_km"):
        original_per_unit = (
            original.line[column] * original.line["length_km"] / original_z_base
        )
        converted_per_unit = (
            converted.line[column] * converted.line["length_km"] / converted_z_base
        )
        np.testing.assert_allclose(converted_per_unit, original_per_unit)


def test_global_seed_repeats_python_numpy_and_torch() -> None:
    set_global_seed(17)
    first = (random.random(), np.random.random(), torch.rand(3))
    set_global_seed(17)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_noncontiguous_reversed_radial_network_is_supported() -> None:
    net = pp.create_empty_network(sn_mva=1.0)
    for bus in (10, 20, 30):
        pp.create_bus(net, vn_kv=12.66, index=bus)
    pp.create_ext_grid(net, bus=10)
    pp.create_line_from_parameters(
        net, 20, 10, 1.0, 0.1, 0.05, 0.4, 1.0
    )
    pp.create_line_from_parameters(
        net, 30, 20, 1.0, 0.1, 0.05, 0.4, 1.0
    )
    pp.create_load(net, bus=30, p_mw=0.2, q_mvar=0.05)
    set_global_seed(5)
    constants = zero_solver.extract_constants({"mg_buses": [30]}, net, 24)
    assert constants["bus_ids"] == [10, 20, 30]
    assert constants["slack"] == 0
    assert constants["from_bus"] == [0, 1]
    assert constants["to_bus"] == [1, 2]
    assert constants["mg_buses"] == [2]
    assert np.all(constants["p_passive"][2] > 0)


class _CustomFollower(FollowerProblem):
    scenario = "BG"

    def __init__(self, T: int) -> None:
        self.T = T

    def lower_solve(self, pi: np.ndarray) -> dict:
        return {"p_mg": np.zeros(self.T), "scenario": self.scenario}


def test_custom_follower_scenario_can_be_registered() -> None:
    def factory(T: int, net=None, **kwargs) -> _CustomFollower:
        return _CustomFollower(T)

    follower_solver.register_follower_scenario("BG", factory, replace=True)
    try:
        solver = follower_solver.build_solver("BG", T=12)
        result = solver.lower_solve(np.ones(12))
        assert result["p_mg"].shape == (12,)
        assert result["scenario"] == "BG"
    finally:
        follower_solver.register_follower_scenario(
            "BG", follower_solver.BuildingGroupSolver, replace=True
        )


class _ScaledFollower(FollowerProblem):
    scenario = "TEST"

    def __init__(self, T: int, scale: float) -> None:
        self.T = T
        self.scale = scale

    def lower_solve(self, pi: np.ndarray) -> dict:
        return {"p_mg": self.scale * np.asarray(pi)}


def test_composite_follower_preserves_one_to_one_price_mapping() -> None:
    composite = CompositeFollowerProblem(
        [_ScaledFollower(3, 1.0), _ScaledFollower(3, 2.0)], T=3
    )
    prices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = composite.lower_solve(prices.reshape(-1))
    np.testing.assert_allclose(result["p_mg"][0], prices[0])
    np.testing.assert_allclose(result["p_mg"][1], 2.0 * prices[1])


class _ScaleMap(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


def test_composite_surrogate_uses_each_model_and_keeps_gradients() -> None:
    surrogate = CompositeGlobalSurrogate(
        [GlobalSurrogate(_ScaleMap(1.0)), GlobalSurrogate(_ScaleMap(3.0))],
        T=3,
    )
    prices = torch.arange(1.0, 7.0, requires_grad=True)
    prediction = surrogate.predict_torch(prices)
    expected = torch.tensor([1.0, 2.0, 3.0, 12.0, 15.0, 18.0])
    torch.testing.assert_close(prediction, expected)
    prediction.sum().backward()
    torch.testing.assert_close(
        prices.grad, torch.tensor([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
    )


def test_independent_microgrid_prices_use_corresponding_revenue() -> None:
    model = SimpleNamespace(
        T=2,
        n_bus=2,
        n_line=0,
        slack=0,
        mg_buses=[0, 1],
        s_base_mva=1.0,
        device="cpu",
        lambda_g=torch.zeros(2),
        p0=torch.zeros(2),
        q0=torch.zeros(2),
        p0_max=10.0,
        q0_max=10.0,
        v_ref=1.0,
        v_min=0.8,
        v_max=1.05,
        p_passive=torch.zeros((2, 2)),
        q_passive=torch.zeros((2, 2)),
        q_injection=torch.zeros((2, 2)),
        v=torch.ones((2, 2)),
        l=torch.empty((0, 2)),
        p=torch.empty((0, 2)),
        q=torch.empty((0, 2)),
        l_max=torch.empty(0),
        r=torch.empty(0),
        x=torch.empty(0),
        from_bus=torch.empty(0, dtype=torch.long),
        to_bus=torch.empty(0, dtype=torch.long),
        child_edges={0: [], 1: []},
        parent_edge={0: None, 1: None},
    )
    prices = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    responses = torch.tensor([[0.5, 1.0], [2.0, 3.0]])
    _, aux = torch_penalty_distflow(
        model,
        prices,
        responses,
        {"eq": 0.0, "ineq": 0.0, "soc": 0.0, "bound": 0.0},
        square_voltage_limits=False,
    )
    expected_revenue = torch.sum(prices * responses)
    torch.testing.assert_close(aux["obj"], -expected_revenue)


def test_torch_distflow_penalizes_reactive_injection_definition() -> None:
    model = SimpleNamespace(
        T=1,
        n_bus=1,
        n_line=0,
        slack=0,
        mg_buses=[],
        s_base_mva=1.0,
        device="cpu",
        lambda_g=torch.zeros(1),
        p0=torch.zeros(1),
        q0=torch.tensor([2.0]),
        p0_max=10.0,
        q0_max=10.0,
        v_ref=1.0,
        v_min=0.8,
        v_max=1.05,
        p_passive=torch.zeros((1, 1)),
        q_passive=torch.zeros((1, 1)),
        q_injection=torch.tensor([[2.0]]),
        v=torch.ones((1, 1)),
        l=torch.empty((0, 1)),
        p=torch.empty((0, 1)),
        q=torch.empty((0, 1)),
        l_max=torch.empty(0),
        r=torch.empty(0),
        x=torch.empty(0),
        from_bus=torch.empty(0, dtype=torch.long),
        to_bus=torch.empty(0, dtype=torch.long),
        child_edges={0: []},
        parent_edge={0: None},
    )

    _, aux = torch_penalty_distflow(
        model,
        torch.zeros(1),
        torch.zeros(1),
        {"eq": 1.0, "ineq": 0.0, "soc": 0.0, "bound": 0.0},
        square_voltage_limits=False,
    )

    torch.testing.assert_close(aux["eq_pen"], torch.tensor(4.0))


def test_multi_experiment_rejects_misaligned_configuration() -> None:
    config = TrustRegionExperimentConfig(
        upper_net=pn.case33bw(),
        lower_nets=[pn.create_cigre_network_lv()],
        mg_buses=[18, 22],
        scenarios=["SB", "SB"],
        map_nets=[_ScaleMap(1.0), _ScaleMap(1.0)],
        save_results=False,
    )
    with pytest.raises(ValueError, match="same nonzero length"):
        build_trust_region_experiment(config)


def test_lower_constants_match_legacy() -> None:
    net = pn.create_cigre_network_lv()
    _assert_dict_equal(
        zero_solver_old.extract_lower_constants({}, net, 24),
        zero_solver.extract_lower_constants({}, net, 24),
    )
    _assert_dict_equal(
        trust_region_solver_old.extract_lower_constants({}, net, 24),
        trust_region_solver.extract_lower_constants({}, net, 24),
    )


class _Map(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.2 + 0.1 * x


def test_nn_initializes_optimized_reactive_injection_from_passive_profile() -> None:
    net = pn.case33bw()
    np.random.seed(9)
    constants = NN_solver.extract_constants({"mg_buses": [18]}, net, 24)
    model = NN_solver.TorchDistFlowOptimizer(constants, _Map())

    assert "q_injection" in dict(model.named_parameters())
    assert "q_injection" not in dict(model.named_buffers())
    torch.testing.assert_close(model.q_injection, model.q_passive)


def test_nn_optimizer_initializes_prices_from_x0() -> None:
    constants = NN_solver.extract_constants(
        {"mg_buses": [18], "pi_min": 0.0, "pi_max": 1.5},
        pn.case33bw(),
        4,
    )
    x0 = np.array([0.1, 0.4, 0.9, 1.4])

    model = NN_solver.TorchDistFlowOptimizer(constants, _Map(), x0=x0)

    np.testing.assert_allclose(
        model.bounded_pi().detach().cpu().numpy(), x0, atol=1e-6
    )


def test_nn_optimizer_rejects_invalid_x0() -> None:
    constants = NN_solver.extract_constants(
        {"mg_buses": [18], "pi_min": 0.0, "pi_max": 1.5},
        pn.case33bw(),
        4,
    )

    with pytest.raises(ValueError, match="must contain 4 prices"):
        NN_solver.TorchDistFlowOptimizer(constants, _Map(), x0=np.ones(3))
    with pytest.raises(ValueError, match="must lie in"):
        NN_solver.TorchDistFlowOptimizer(
            constants, _Map(), x0=np.array([0.1, 0.4, 0.9, 1.6])
        )


def test_nn_experiment_uses_multi_follower_x0() -> None:
    x0 = np.array(
        [[0.1, 0.4, 0.9, 1.4], [0.2, 0.5, 1.0, 1.3]], dtype=float
    )
    result = NN_solver.run_nn_solver_experiment(
        NN_solver.NNSolverExperimentConfig(
            upper_net=pn.case33bw(),
            lower_nets=[
                pn.create_cigre_network_lv(),
                pn.create_cigre_network_lv(),
            ],
            mg_buses=[18, 22],
            scenarios=["SB1", "SB2"],
            T=4,
            x0=x0,
            steps=0,
            map_nets=[_Map(), _Map()],
            evaluate_true=False,
            save_results=False,
        )
    )

    np.testing.assert_allclose(result["optimal pi"], x0, atol=1e-6)


def test_tr_initializes_optimized_reactive_injection_from_passive_profile() -> None:
    net = pn.case33bw()
    np.random.seed(10)
    constants = trust_region_solver.extract_constants(
        {"mg_buses": [18]}, net, 24
    )
    model = trust_region_solver._LeaderSubproblem(
        constants, trust_region_solver.GlobalSurrogate(_Map())
    )

    assert "q_injection" in dict(model.named_parameters())
    assert "q_injection" not in dict(model.named_buffers())
    torch.testing.assert_close(model.q_injection, model.q_passive)


def test_load_data_concatenates_original_and_tagged_samples(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(L2Otraining, "DATA_DIR", tmp_path)
    x_orig = np.ones((3, 24), dtype=np.float32)
    y_orig = np.full((3, 24), 2.0, dtype=np.float32)
    x_tr = np.full((2, 24), 3.0, dtype=np.float32)
    y_tr = np.full((2, 24), 4.0, dtype=np.float32)
    np.save(tmp_path / "X_BG.npy", x_orig)
    np.save(tmp_path / "Y_BG.npy", y_orig)
    np.save(tmp_path / "X_BG_tr.npy", x_tr)
    np.save(tmp_path / "Y_BG_tr.npy", y_tr)

    x_only, y_only = L2Otraining.load_data("BG", data_tag="tr")
    assert tuple(x_only.shape) == (2, 24)
    np.testing.assert_allclose(x_only.numpy(), x_tr)
    np.testing.assert_allclose(y_only.numpy(), y_tr)

    x_all, y_all = L2Otraining.load_data("BG", data_tag="tr", include_original=True)
    assert tuple(x_all.shape) == (5, 24)
    assert tuple(y_all.shape) == (5, 24)
    np.testing.assert_allclose(x_all[:3].numpy(), x_orig)
    np.testing.assert_allclose(x_all[3:].numpy(), x_tr)
    np.testing.assert_allclose(y_all[:3].numpy(), y_orig)
    np.testing.assert_allclose(y_all[3:].numpy(), y_tr)


def test_instance_training_data_uses_exact_file_names(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(L2Otraining, "DATA_DIR", tmp_path)
    x = np.ones((3, 24), dtype=np.float32)
    y = np.full((3, 24), 2.0, dtype=np.float32)
    np.save(tmp_path / "X_SB.npy", x)
    np.save(tmp_path / "Y_SB.npy", y)

    with pytest.raises(FileNotFoundError, match="X_SB1.npy"):
        L2Otraining.load_data("SB1")

    np.save(tmp_path / "X_SB1.npy", x)
    np.save(tmp_path / "Y_SB1.npy", y)
    x_loaded, y_loaded = L2Otraining.load_data("SB1")
    np.testing.assert_allclose(x_loaded.numpy(), x)
    np.testing.assert_allclose(y_loaded.numpy(), y)


def test_data_generation_reuses_exact_instance_dataset(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(data_generation, "DATA_DIR", tmp_path)
    x = np.arange(72, dtype=float).reshape(3, 24)
    y = x + 1.0
    np.save(tmp_path / "X_SB1.npy", x)
    np.save(tmp_path / "Y_SB1.npy", y)

    module = types.ModuleType("must_not_build")

    def fail_build_solver(**kwargs):
        raise AssertionError("Existing complete data should bypass solver creation.")

    module.build_solver = fail_build_solver
    x_loaded, y_loaded = data_generation.data_generation(
        24,
        3,
        "SB1",
        solver_module=module,
        reuse_existing=True,
    )

    np.testing.assert_array_equal(x_loaded, x)
    np.testing.assert_array_equal(y_loaded, y)


def test_l2o_loader_matches_training_metadata() -> None:
    metadata_path = L2Otraining.MODEL_DIR / "L2O_SB.json"
    checkpoint_path = L2Otraining.MODEL_DIR / "L2O_SB.pth"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        return
    model = NN_solver.load_l2o_map_net("SB", device=torch.device("cpu"))
    x_data, _ = L2Otraining.load_data("SB")
    with torch.no_grad():
        prediction = model(x_data[:2])
    assert prediction.shape == (2, x_data.shape[1])
    assert torch.isfinite(prediction).all()


def test_follower_and_data_generation_match_legacy(tmp_path, monkeypatch) -> None:
    price = np.linspace(0.2, 1.2, 24)
    old_result = follower_solver_old.build_solver("SB", T=24).lower_solve(price)
    new_result = follower_solver.build_solver("SB", T=24).lower_solve(price)
    np.testing.assert_allclose(old_result["p_mg"], new_result["p_mg"])
    np.testing.assert_allclose(old_result["objective"], new_result["objective"])

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(PROJECT_ROOT / "data" / "DA_RT_data.csv", data_dir)
    monkeypatch.setattr(data_generation, "DATA_DIR", data_dir)
    old_x, old_y = data_generation.data_generation(
        24, 2, "SB", solver_module=follower_solver_old, seed=42
    )
    new_x, new_y = data_generation.data_generation(
        24, 2, "SB", solver_module=follower_solver, seed=42
    )
    np.testing.assert_array_equal(old_x, new_x)
    np.testing.assert_allclose(old_y, new_y)


def test_hard_distflow_enforces_reactive_injection_definition() -> None:
    net = pn.case33bw()
    pi = np.full(24, 0.5)
    p_mg = np.full(24, 0.1)

    np.random.seed(12)
    new_zero = zero_solver.extract_constants({"mg_buses": [18]}, net, 24)
    zero_leader = zero_solver.UpperSolver(new_zero)

    np.random.seed(13)
    new_tr = trust_region_solver.extract_constants({"mg_buses": [18]}, net, 24)
    tr_leader = trust_region_solver.DistributionNetworkLeader(new_tr)

    for evaluator, constants, solve in (
        (zero_leader.distflow_evaluator, new_zero, zero_leader.distflow),
        (tr_leader.distflow_evaluator, new_tr, tr_leader._solve_power_flow),
    ):
        constraint_names = {
            constraint.getName() for constraint in evaluator.model.getConstrs()
        }
        q_definition_names = {
            name for name in constraint_names if name.startswith("q_inj_def")
        }
        assert len(q_definition_names) == constants["n_bus"] * constants["T"]
        assert np.isfinite(solve(pi, p_mg))
        q_injection = np.array(
            [
                [evaluator.q_injection[bus, t].x for t in range(constants["T"])]
                for bus in range(constants["n_bus"])
            ]
        )
        np.testing.assert_allclose(q_injection, constants["q_passive"])


def test_nn_and_tr_optimize_reactive_injection_but_keep_passive_profile_fixed() -> None:
    net = pn.case33bw()

    np.random.seed(14)
    nn_constants = NN_solver.extract_constants({"mg_buses": [18]}, net, 24)
    nn_model = NN_solver.TorchDistFlowOptimizer(nn_constants, _Map())

    np.random.seed(14)
    tr_constants = trust_region_solver.extract_constants({"mg_buses": [18]}, net, 24)
    tr_model = trust_region_solver._LeaderSubproblem(
        tr_constants, trust_region_solver.GlobalSurrogate(_Map())
    )

    for model, constants in (
        (nn_model, nn_constants),
        (tr_model, tr_constants),
    ):
        assert "q_passive" not in dict(model.named_parameters())
        assert "q_passive" in dict(model.named_buffers())
        assert "q_injection" in dict(model.named_parameters())
        np.testing.assert_allclose(
            model.q_passive.detach().cpu().numpy(), constants["q_passive"]
        )
        torch.testing.assert_close(model.q_injection, model.q_passive)

        passive_before = model.q_passive.detach().clone()
        injection_before = model.q_injection.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        assert model.q_injection.grad is not None
        assert torch.linalg.vector_norm(model.q_injection.grad) > 0
        optimizer.step()
        torch.testing.assert_close(model.q_passive, passive_before)
        assert not torch.equal(model.q_injection, injection_before)


class _RecordingFollower(FollowerProblem):
    scenario = "TEST"

    def __init__(self, T: int, scale: float = 1.0) -> None:
        self.T = T
        self.scale = scale
        self.calls: list[np.ndarray] = []

    def lower_solve(self, pi: np.ndarray) -> dict:
        prices = np.asarray(pi, dtype=float).reshape(self.T)
        self.calls.append(prices.copy())
        return {"p_mg": self.scale * prices}


class _TrustRegionStubLeader:
    def __init__(self, x0: np.ndarray) -> None:
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.solve_calls: list[dict] = []

    def get_feasible_set(self) -> dict:
        return {
            "lb": np.zeros_like(self.x0),
            "ub": np.ones_like(self.x0),
        }

    def solve(self, surrogate, trust_region=None, x_center=None, solve_params=None) -> tuple:
        center = self.x0 if x_center is None else np.asarray(x_center, dtype=float)
        candidate = center.copy()
        radius = 0.2 if trust_region is None else trust_region.get_radius()
        candidate[0] += min(0.05, 0.5 * radius)
        self.solve_calls.append({
            "trust_region": trust_region,
            "x_center": center.copy(),
            "steps": None if solve_params is None else solve_params.get("steps"),
        })
        return candidate, 0.0

    def evaluate_surrogate_objective(self, x: np.ndarray, surrogate) -> float:
        return 1.0 if np.allclose(x, self.x0) else 0.0

    def evaluate_objective(self, x: np.ndarray, y: np.ndarray) -> float:
        return 0.0 if np.allclose(x, self.x0) else 1.0


@pytest.mark.parametrize(
    ("model_type", "rbf_kernel"),
    [
        pytest.param("linear", None, id="affine"),
        pytest.param("rbf", "linear", id="rbf-linear"),
        pytest.param("rbf", "cubic", id="rbf-cubic"),
        pytest.param("rbf", "thin_plate_spline", id="rbf-thin-plate"),
        pytest.param("rbf", "gaussian", id="rbf-gaussian"),
        pytest.param("rbf", "multiquadric", id="rbf-multiquadric"),
        pytest.param(
            "rbf", "inverse_multiquadric", id="rbf-inverse-multiquadric"
        ),
    ],
)
def test_residual_model_experiment_interface(
    model_type: str, rbf_kernel: str | None
) -> None:
    center = np.array([0.4, 0.6])
    X = np.array([
        center,
        [0.2, 0.6],
        [0.6, 0.6],
        [0.4, 0.4],
        [0.4, 0.8],
        [0.2, 0.4],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.6, 0.8],
    ])
    y = np.column_stack([
        np.sin(X[:, 0]) + X[:, 1] ** 2,
        X[:, 0] * X[:, 1] + 0.1 * X[:, 0],
    ])
    kwargs = {"model_type": model_type}
    if rbf_kernel is not None:
        kwargs.update({
            "rbf_kernel": rbf_kernel,
            "rbf_smoothing": 1e-5,
            "rbf_degree": 1,
            "rbf_scale": "trust_radius",
            "rbf_epsilon": 1.0,
        })
    model = ResidualModel(**kwargs)
    model.fit(X, y, x_center=center, radius=0.2)

    query = np.array([0.45, 0.55])
    prediction = model.predict(query)
    assert prediction.shape == (2,)
    assert np.all(np.isfinite(prediction))

    query_t = torch.tensor(query, dtype=torch.float32, requires_grad=True)
    torch_prediction = model.predict_torch(query_t)
    torch_prediction.sum().backward()
    assert torch_prediction.shape == (2,)
    assert torch.all(torch.isfinite(torch_prediction))
    assert query_t.grad is not None
    assert torch.all(torch.isfinite(query_t.grad))


def _build_lightweight_tr_controller(
    follower: FollowerProblem,
    *,
    T: int = 2,
    n_followers: int = 1,
    n_samples: int = 2,
    save_true_responses: bool = True,
    rbf_degree: int = 0,
    rbf_smoothing: float = 1e-6,
    rbf_geometry_condition_max: float = 1e12,
    rbf_geometry_max_attempts: int = 6,
    residual_update_mode: str = "replace",
    radius_policy: str = "adaptive",
    acceptance_policy: str = "filter",
    fixed_radius: float = 0.05,
    max_consecutive_rejections: int = 10,
) -> tuple[TRResidualController, np.ndarray]:
    x0 = np.full(n_followers * T, 0.5)
    if n_followers == 1:
        surrogate = GlobalSurrogate(_ScaleMap(0.0))
    else:
        surrogate = CompositeGlobalSurrogate(
            [GlobalSurrogate(_ScaleMap(0.0)) for _ in range(n_followers)],
            T=T,
        )
    controller = TRResidualController(
        leader=_TrustRegionStubLeader(x0),
        follower=follower,
        global_surrogate=surrogate,
        residual_model=ResidualModel(
            model_type="rbf",
            rbf_kernel="cubic",
            rbf_smoothing=rbf_smoothing,
            rbf_degree=rbf_degree,
            rbf_scale="trust_radius",
        ),
        tr_params={
            "R_0": 0.2,
            "R_max": 1.0,
            "gamma_dec": 0.7,
            "gamma_inc": 1.5,
            "eta_1": 0.1,
            "eta_2": 0.9,
        },
        algo_params={
            "n_samples": n_samples,
            "K_max": 2,
            "x0": x0,
            "eps_x": 1e-12,
            "eps_f": 1e-12,
            "eps_R": 1e-12,
            "sampling_method": "lhs",
            "residual_update_mode": residual_update_mode,
            "radius_policy": radius_policy,
            "acceptance_policy": acceptance_policy,
            "fixed_radius": fixed_radius,
            "max_consecutive_rejections": max_consecutive_rejections,
            "rbf_geometry_condition_max": rbf_geometry_condition_max,
            "rbf_geometry_max_attempts": rbf_geometry_max_attempts,
            "random_seed": 7,
            "save_true_responses": save_true_responses,
            "solve_params": {
                "steps_init": 1,
                "steps_warm": 1,
                "lr": 1e-2,
            },
        },
    )
    return controller, x0


def test_tr_resamples_after_rejected_radius_shrink() -> None:
    follower = _RecordingFollower(T=2)
    controller, x0 = _build_lightweight_tr_controller(follower)

    first, x_next, _ = controller._one_iteration(0, x0)
    assert first["decision"] == "reject"
    first_call_count = len(follower.calls)

    second, _, _ = controller._one_iteration(1, x_next)
    assert second["decision"] == "reject"
    np.testing.assert_allclose(second["R"], 0.7 * 0.7 * 0.2)
    assert first_call_count == controller.n_samples + 1
    assert len(follower.calls) - first_call_count == controller.n_samples + 1


def test_tr_accepts_true_improvement_and_shrinks_for_bad_ratio() -> None:
    follower = _RecordingFollower(T=2)
    controller, x0 = _build_lightweight_tr_controller(follower)
    controller.leader.evaluate_surrogate_objective = (
        lambda x, surrogate: 0.0 if np.allclose(x, x0) else 1.0
    )
    controller.leader.evaluate_objective = (
        lambda x, y: 1.0 if np.allclose(x, x0) else 0.0
    )

    record, x_next, converged = controller._one_iteration(0, x0)

    assert record["decision"] == "accept_shrink"
    assert record["rho"] == -np.inf
    assert record["R"] == pytest.approx(0.14)
    np.testing.assert_allclose(x_next, record["x_cand"])
    assert record["n_residuals"] == 1
    assert converged is False


@pytest.mark.parametrize(
    ("true_current", "true_candidate", "pred_current", "pred_candidate", "expected"),
    [
        (0.0, 1.0, 1.0, 0.0, "reject"),
        (1.0, 0.0, 0.0, 1.0, "accept_shrink"),
        (1.0, 0.5, 1.0, 0.0, "accept_keep"),
        (2.0, 0.5, 1.0, 0.0, "accept_expand"),
    ],
)
def test_fixed_radius_preserves_filter_decisions(
    true_current: float,
    true_candidate: float,
    pred_current: float,
    pred_candidate: float,
    expected: str,
) -> None:
    controller, x0 = _build_lightweight_tr_controller(
        _RecordingFollower(T=2), radius_policy="fixed"
    )
    controller.leader.evaluate_objective = (
        lambda x, y: true_current if np.allclose(x, x0) else true_candidate
    )
    controller.leader.evaluate_surrogate_objective = (
        lambda x, surrogate: pred_current
        if np.allclose(x, x0) else pred_candidate
    )

    record, x_next, _ = controller._one_iteration(0, x0)

    assert record["decision"] == record["would_be_decision"] == expected
    assert record["R_before"] == pytest.approx(0.05)
    assert record["R_after"] == pytest.approx(0.05)
    assert record["R"] == pytest.approx(0.05)
    assert record["residual_committed"] is (expected != "reject")
    np.testing.assert_allclose(
        x_next, x0 if expected == "reject" else record["x_cand"]
    )


@pytest.mark.parametrize("pred_current,pred_candidate", [(1.0, 0.0), (0.0, 1.0)])
@pytest.mark.parametrize("radius_policy", ["adaptive", "fixed"])
def test_always_accept_commits_worsening_candidate(
    pred_current: float, pred_candidate: float, radius_policy: str
) -> None:
    controller, x0 = _build_lightweight_tr_controller(
        _RecordingFollower(T=2),
        radius_policy=radius_policy,
        acceptance_policy="always_accept",
    )
    controller.leader.evaluate_surrogate_objective = (
        lambda x, surrogate: pred_current
        if np.allclose(x, x0) else pred_candidate
    )

    record, x_next, _ = controller._one_iteration(0, x0)

    assert record["would_be_decision"] == "reject"
    assert record["decision"] == "accept_forced_shrink"
    assert record["residual_committed"] is True
    assert record["consecutive_rejections"] == 0
    assert record["n_residuals"] == 1
    assert record["F_next"] == record["F_candidate"]
    assert record["composite_queries"] == controller.n_samples + 1
    np.testing.assert_allclose(x_next, record["x_cand"])
    assert record["R_after"] == pytest.approx(
        0.14 if radius_policy == "adaptive" else 0.05
    )


@pytest.mark.parametrize("radius_policy", ["adaptive", "fixed"])
@pytest.mark.parametrize("acceptance_policy", ["filter", "always_accept"])
def test_two_iteration_ablation_smoke(
    radius_policy: str, acceptance_policy: str
) -> None:
    follower = _RecordingFollower(T=2)
    controller, _ = _build_lightweight_tr_controller(
        follower,
        n_samples=60,
        radius_policy=radius_policy,
        acceptance_policy=acceptance_policy,
        residual_update_mode="accumulate",
    )
    controller.leader.evaluate_objective = lambda x, y: float(x[0])
    controller.leader.evaluate_surrogate_objective = (
        lambda x, surrogate: float(-x[0])
    )

    result = controller.run()

    assert result["n_iter"] == 2
    assert len(follower.calls) == 2 * (controller.n_samples + 1)
    assert result["history"][-1]["cumulative_composite_queries"] == len(follower.calls)
    expected_residuals = 0 if acceptance_policy == "filter" else 2
    assert result["history"][-1]["n_residuals"] == expected_residuals
    assert result["history"][-1]["R_after"] == pytest.approx(
        0.2 * 0.7**2 if radius_policy == "adaptive" else 0.05
    )


def test_fixed_filter_stops_after_ten_consecutive_rejections() -> None:
    follower = _RecordingFollower(T=2)
    controller, _ = _build_lightweight_tr_controller(
        follower, radius_policy="fixed"
    )
    controller.K_max = 12

    result = controller.run()

    assert result["n_iter"] == 10
    assert result["converged"] is False
    assert result["stop_reason"] == "consecutive_rejection_limit"
    assert [row["consecutive_rejections"] for row in result["history"]] == list(range(1, 11))
    assert all(row["decision"] == "reject" for row in result["history"])
    assert result["history"][-1]["stop_reason"] == "consecutive_rejection_limit"
    assert len(follower.calls) == 10 * (controller.n_samples + 1)


def test_fixed_filter_acceptance_resets_rejection_streak() -> None:
    controller, x0 = _build_lightweight_tr_controller(
        _RecordingFollower(T=2), radius_policy="fixed"
    )
    first, _, _ = controller._one_iteration(0, x0)
    assert first["consecutive_rejections"] == 1
    controller.leader.evaluate_objective = (
        lambda x, y: 1.0 if np.allclose(x, x0) else 0.0
    )

    second, x_next, _ = controller._one_iteration(1, x0)

    assert second["decision"].startswith("accept")
    assert second["consecutive_rejections"] == 0
    third, _, _ = controller._one_iteration(2, x_next)
    assert third["decision"] == "reject"
    assert third["consecutive_rejections"] == 1


def _constant_residual(value: float) -> ResidualModel:
    model = ResidualModel(model_type="linear")
    model.fit(
        np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]]),
        np.full((3, 2), value),
    )
    return model


@pytest.mark.parametrize(
    ("update_mode", "expected"),
    [
        pytest.param("replace", 3.0, id="replace-uses-new-only"),
        pytest.param("accumulate", 4.0, id="accumulate-adds-old-and-new"),
    ],
)
def test_temporary_residual_update_mode(update_mode: str, expected: float) -> None:
    base = GlobalSurrogate(_ScaleMap(0.0))
    base.merge(_constant_residual(1.0))
    temporary = trust_region_module._TempCorrectedSurrogate(
        base,
        _constant_residual(3.0),
        update_mode=update_mode,
    )
    query = np.array([0.4, 0.6])

    np.testing.assert_allclose(temporary.predict(query), expected)
    torch_prediction = temporary.predict_torch(
        torch.tensor(query, dtype=torch.float32)
    )
    torch.testing.assert_close(
        torch_prediction,
        torch.full((2,), expected, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("update_mode", "expected_count"),
    [
        pytest.param("replace", 1, id="replace"),
        pytest.param("accumulate", 2, id="accumulate"),
    ],
)
def test_controller_commits_selected_residual_update_mode(
    update_mode: str, expected_count: int
) -> None:
    controller, _ = _build_lightweight_tr_controller(
        _RecordingFollower(T=2), residual_update_mode=update_mode
    )
    controller.global_surrogate.merge(_constant_residual(1.0))

    controller._commit_residual_update(_constant_residual(2.0))

    assert len(controller.global_surrogate._residuals) == expected_count
    np.testing.assert_allclose(
        controller.global_surrogate.predict(np.array([0.4, 0.6])),
        2.0 if update_mode == "replace" else 3.0,
    )


def test_controller_rejects_unknown_residual_update_mode() -> None:
    with pytest.raises(ValueError, match="residual_update_mode"):
        _build_lightweight_tr_controller(
            _RecordingFollower(T=2), residual_update_mode="rolling"
        )


def test_tr_refill_switches_from_duplicate_star_points_to_lhs(monkeypatch) -> None:
    controller, x0 = _build_lightweight_tr_controller(
        _RecordingFollower(T=2), n_samples=2
    )
    include_star_history = []

    def fake_sample_in_trust_region(
        x_center,
        tr,
        n,
        feasible_set,
        method,
        seed,
        include_star_points,
        max_star_points=None,
    ):
        include_star_history.append(include_star_points)
        center = np.asarray(x_center, dtype=float)
        if include_star_points:
            return np.repeat(center.reshape(1, -1), n, axis=0)
        return np.asarray(
            [center + np.array([0.01 * (index + 1), 0.0]) for index in range(n)]
        )

    monkeypatch.setattr(
        trust_region_module,
        "sample_in_trust_region",
        fake_sample_in_trust_region,
    )
    rows = controller._sample_composite_rows(
        0,
        x0,
        controller.leader.get_feasible_set(),
        missing_counts=[2],
        attempt=0,
    )

    assert rows.shape == (2, 2)
    assert include_star_history == [True, False]
    assert np.all(np.max(np.abs(rows - x0), axis=1) > 0.0)


def test_tr_small_refill_mixes_star_and_lhs_candidate_points(monkeypatch) -> None:
    controller, x0 = _build_lightweight_tr_controller(
        _RecordingFollower(T=2), n_samples=8
    )
    sample_calls = []

    def fake_sample_in_trust_region(
        x_center,
        tr,
        n,
        feasible_set,
        method,
        seed,
        include_star_points,
        max_star_points=None,
    ):
        sample_calls.append((n, include_star_points, max_star_points, seed))
        center = np.asarray(x_center, dtype=float)
        points = []
        if include_star_points and max_star_points:
            points.append(center + np.array([0.08, 0.0]))
        while len(points) < n:
            index = len(points) + 1
            points.append(
                center
                + np.array([
                    -0.09 + 0.01 * index,
                    0.02 + 0.005 * index,
                ])
            )
        return np.asarray(points)

    monkeypatch.setattr(
        trust_region_module,
        "sample_in_trust_region",
        fake_sample_in_trust_region,
    )
    rows = controller._sample_composite_rows(
        0,
        x0,
        controller.leader.get_feasible_set(),
        missing_counts=[2],
        attempt=0,
    )

    assert rows.shape == (2, 2)
    assert sample_calls[0][0] == 2
    assert sample_calls[0][1:3] == (True, 2)
    assert np.count_nonzero(np.abs(rows[:, 1] - x0[1]) > 1e-12) >= 1


def test_tr_24d_design_uses_full_coordinate_star_then_lhs() -> None:
    controller, x0 = _build_lightweight_tr_controller(
        _RecordingFollower(T=24), T=24, n_samples=60, rbf_degree=1
    )

    rows = controller._sample_composite_rows(
        0,
        x0,
        controller.leader.get_feasible_set(),
        missing_counts=[controller.n_samples - 1],
        attempt=0,
    )

    expected_star = []
    for index in range(24):
        direction = np.zeros(24)
        direction[index] = controller.tr.get_radius()
        expected_star.extend([x0 + direction, x0 - direction])
    np.testing.assert_allclose(rows[:48], np.asarray(expected_star))
    assert rows.shape == (59, 24)
    assert np.unique(rows, axis=0).shape[0] == 59


def test_nested_star_shell_geometry_is_rejected_before_fitting() -> None:
    d = 4
    controller, x0 = _build_lightweight_tr_controller(
        _RecordingFollower(T=d),
        T=d,
        n_samples=4 * d,
        rbf_degree=1,
        rbf_smoothing=0.0,
        rbf_geometry_condition_max=1e12,
    )
    points = [x0]
    for radius in (0.1, 0.1 - 2e-8):
        for index in range(d):
            direction = np.zeros(d)
            direction[index] = radius
            points.extend([x0 + direction, x0 - direction])
    nested_design = np.asarray(points)

    metrics = controller._rbf_design_quality(
        nested_design, x0, trust_radius=0.3
    )

    assert metrics["polynomial_rank"] == d + 1
    assert metrics["condition_number"] > controller.rbf_geometry_condition_max
    assert metrics["is_valid"] is False


def test_degree_one_geometry_is_full_rank_and_bounded_without_extra_queries() -> None:
    follower = _RecordingFollower(T=3)
    controller, x0 = _build_lightweight_tr_controller(
        follower,
        T=3,
        n_samples=6,
        rbf_degree=1,
        rbf_geometry_condition_max=1e12,
    )

    _, fit_data = controller._collect_residual_fit_data(
        0,
        x0,
        controller.leader.get_feasible_set(),
        trust_radius=0.3,
    )
    quality = controller._last_rbf_geometry_metrics[0]

    assert len(fit_data[0][0]) == controller.n_samples
    assert quality["polynomial_rank"] == x0.size + 1
    assert quality["condition_number"] <= controller.rbf_geometry_condition_max
    assert len(follower.calls) == controller.n_samples
    fitted = ResidualModel(**controller.residual_model_kwargs)
    fitted.fit(
        fit_data[0][0],
        np.zeros_like(fit_data[0][1]),
        x_center=x0,
        radius=0.3,
    )
    assert fitted.rbf_condition == pytest.approx(
        quality["condition_number"], rel=1e-12
    )


def test_geometry_failure_precedes_residual_lower_solve() -> None:
    follower = _RecordingFollower(T=3)
    controller, x0 = _build_lightweight_tr_controller(
        follower,
        T=3,
        n_samples=6,
        rbf_degree=1,
        rbf_geometry_condition_max=1.0,
        rbf_geometry_max_attempts=3,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"follower_index=0, missing_count=5, polynomial_rank=\d+, "
            r"condition_number=.*attempt_count=3"
        ),
    ):
        controller._collect_residual_fit_data(
            0,
            x0,
            controller.leader.get_feasible_set(),
            trust_radius=0.3,
        )

    assert len(follower.calls) == 0


def test_tr_resamples_geometry_after_multiple_radius_shrinks() -> None:
    follower = _RecordingFollower(T=2)
    controller, x_k = _build_lightweight_tr_controller(follower, n_samples=6)
    records = []
    for k in range(5):
        record, x_k, _ = controller._one_iteration(k, x_k)
        records.append(record)

    assert all(record["decision"] == "reject" for record in records)
    assert len(follower.calls) == 5 * (controller.n_samples + 1)
    assert all(
        record["max_rbf_condition"]
        <= controller.rbf_geometry_condition_max
        for record in records
    )
    assert all("geometry_resample_attempts" in record for record in records)


def test_multi_follower_fresh_samples_use_one_composite_batch() -> None:
    child_followers = [_RecordingFollower(T=2), _RecordingFollower(T=2, scale=2.0)]
    composite = CompositeFollowerProblem(child_followers, T=2)
    controller, x0 = _build_lightweight_tr_controller(
        composite, T=2, n_followers=2, n_samples=2
    )

    _, fit_data = controller._collect_residual_fit_data(
        0,
        x0,
        controller.leader.get_feasible_set(),
        trust_radius=0.3,
    )

    assert [len(values) for values, _ in fit_data] == [2, 2]
    assert composite.individual_follower_solve_count == 4
    assert [len(follower.calls) for follower in child_followers] == [2, 2]


def test_tr_run_uses_steps_init_then_steps_warm() -> None:
    follower = _RecordingFollower(T=2)
    controller, x0 = _build_lightweight_tr_controller(follower, n_samples=2)
    controller.K_max = 1
    controller.solve_params["steps_init"] = 5
    controller.solve_params["steps_warm"] = 11

    result = controller.run()

    assert [call["steps"] for call in controller.leader.solve_calls] == [5, 11]
    assert controller.leader.solve_calls[0]["trust_region"] is None
    np.testing.assert_allclose(controller.leader.solve_calls[0]["x_center"], x0)
    np.testing.assert_allclose(
        result["x_init"], controller.leader.solve_calls[1]["x_center"]
    )
    assert controller.leader.solve_calls[1]["trust_region"] is not None
    samples = result["true_responses"]["TEST"]
    assert samples["X"].shape[1] == 2
    assert samples["Y"].shape[0] == controller.n_samples


def test_tr_can_run_without_retaining_true_response_samples() -> None:
    follower = _RecordingFollower(T=2)
    controller, _ = _build_lightweight_tr_controller(
        follower, n_samples=2, save_true_responses=False
    )
    controller.K_max = 1

    result = controller.run()

    assert "true_responses" not in result
    assert controller.collected_true_responses() == {}
    assert all(not blocks for blocks in controller._sampled_X)
    assert all(not blocks for blocks in controller._sampled_Y)


def test_tr_true_responses_are_saved_by_scenario(tmp_path) -> None:
    child_followers = [
        _RecordingFollower(T=2),
        _RecordingFollower(T=2, scale=2.0),
        _RecordingFollower(T=2, scale=3.0),
    ]
    child_followers[0].scenario = "SB"
    child_followers[1].scenario = "SB"
    child_followers[2].scenario = "BG"
    composite = CompositeFollowerProblem(child_followers, T=2)
    controller, _ = _build_lightweight_tr_controller(
        composite, T=2, n_followers=3, n_samples=2
    )
    controller.K_max = 1
    controller.scenarios = ["SB", "SB", "BG"]

    result = controller.run()
    original_sb = tmp_path / "X_SB.npy"
    original_sb.write_bytes(b"keep-me")
    paths = save_trust_region_true_responses(
        result["true_responses"], output_dir=tmp_path, tag="tr"
    )

    assert set(result["true_responses"]) == {"SB", "BG"}
    assert result["true_responses"]["SB"]["X"].shape[0] == 2 * controller.n_samples
    assert result["true_responses"]["BG"]["X"].shape[0] == controller.n_samples
    assert paths["SB"]["X"].endswith("X_SB_tr.npy")
    assert paths["BG"]["Y"].endswith("Y_BG_tr.npy")
    assert original_sb.read_bytes() == b"keep-me"
    assert (tmp_path / "X_SB_tr.npy").is_file()
    assert (tmp_path / "Y_SB_tr.npy").is_file()
    assert (tmp_path / "X_BG_tr.npy").is_file()
    assert not (tmp_path / "X_IP_tr.npy").exists()


def test_tr_true_responses_stay_separate_for_named_instances() -> None:
    child_followers = [
        _RecordingFollower(T=2),
        _RecordingFollower(T=2, scale=2.0),
    ]
    composite = CompositeFollowerProblem(
        child_followers, T=2, follower_ids=["SB1", "SB2"]
    )
    controller, _ = _build_lightweight_tr_controller(
        composite, T=2, n_followers=2, n_samples=2
    )
    controller.K_max = 1
    controller.scenarios = ["SB1", "SB2"]

    result = controller.run()

    assert set(result["true_responses"]) == {"SB1", "SB2"}
    assert result["true_responses"]["SB1"]["X"].shape[0] == 2
    assert result["true_responses"]["SB2"]["X"].shape[0] == 2
