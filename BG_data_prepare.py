"""Prepare full-year hourly energy profiles for four buildings.

This is the headless counterpart of ``BG_data_prepare.ipynb``. It converts raw
15-minute data into 2023 hourly CSVs used by ``buildings.py``.

Each output CSV:
    - uses ``timestamp`` as the hourly index
    - contains ``load_kwh``, ``pv_kwh``, and ``outdoor_temperature_c``
    - contains all 8,760 consecutive hours of 2023
    - includes only lighting and electric-equipment loads (HVAC excluded)
    - scales PV by the building-specific weight in ``PV_SCALING_WEIGHTS``

Server example::

    python -u BG_data_prepare.py
    python -u BG_data_prepare.py --plot-date 2023-07-16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_YEAR = 2023
DATA_DIR = PROJECT_ROOT / "building data"
OUTPUT_DIR = PROJECT_ROOT / "data"
TEMPERATURE_FILE = DATA_DIR / "Temperature_2023.csv"
PLOT_DATE = "2023-07-16"

PV_SCALING_WEIGHTS = {
    "Residential 1": 0.5,
    "Residential 2": 0.5,
    "Large Commercial 1": 1.0,
    "Large Commercial 2": 0.25,
}

LOAD_COLUMN_MARKERS = (
    "Lights Electricity Rate [W]",
    "Electric Equipment Electricity Rate [W]",
)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


def expected_hour_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 00:00:00",
        f"{year}-12-31 23:00:00",
        freq="h",
        name="timestamp",
    )


def energyplus_interval_starts(values: pd.Series, year: int) -> pd.Series:
    """Convert EnergyPlus interval-end labels to 15-minute start timestamps."""
    text = values.astype(str).str.strip()
    parts = text.str.extract(
        r"(?P<month>\d{2})/(?P<day>\d{2})\s+"
        r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})$"
    )
    if parts.isna().any().any():
        bad = text.loc[parts.isna().any(axis=1)].head().tolist()
        raise ValueError(f"Cannot parse EnergyPlus timestamps: {bad}")

    calendar_day = pd.to_datetime(
        {
            "year": pd.Series(year, index=parts.index),
            "month": parts["month"].astype(int),
            "day": parts["day"].astype(int),
        }
    )
    end_minutes = parts["hour"].astype(int) * 60 + parts["minute"].astype(int)
    interval_starts = (
        calendar_day
        + pd.to_timedelta(end_minutes, unit="m")
        + pd.to_timedelta(parts["second"].astype(int), unit="s")
        - pd.Timedelta(minutes=15)
    )
    interval_starts.name = "timestamp"
    return interval_starts


def require_four_intervals_per_hour(
    hour_index: pd.Series,
    source_name: str,
    year: int,
) -> None:
    expected = expected_hour_index(year)
    counts = hour_index.value_counts().sort_index().reindex(expected, fill_value=0)
    invalid = counts[counts != 4]
    if not invalid.empty:
        sample = {str(key): int(value) for key, value in invalid.head().items()}
        raise ValueError(
            f"{source_name} must contain four 15-minute records per hour; "
            f"invalid sample: {sample}"
        )


def prepare_hourly_load(path: Path, year: int) -> pd.Series:
    """Extract non-HVAC load and aggregate 15-minute rates to hourly kWh."""
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    source_time = raw.iloc[:, 0]
    timestamps = energyplus_interval_starts(source_time, year)

    load_columns = [
        column
        for column in raw.columns
        if any(marker in column for marker in LOAD_COLUMN_MARKERS)
    ]
    if not load_columns:
        raise ValueError(f"No lights/electric-equipment columns found in {path.name}")

    power_w = raw[load_columns].apply(pd.to_numeric, errors="raise").sum(axis=1)
    interval_kwh = power_w * 0.25 / 1000.0
    hour_index = timestamps.dt.floor("h")
    require_four_intervals_per_hour(hour_index, path.name, year)

    hourly = interval_kwh.groupby(hour_index).sum().reindex(expected_hour_index(year))
    hourly.name = "load_kwh"
    return hourly


def prepare_hourly_pv(path: Path, year: int) -> pd.Series:
    """Select one complete PV year and sum interval generation to hourly kWh."""
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    required = {"Time", "generation(kWh)"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(raw["Time"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")

    selected = raw.loc[timestamps.dt.year == year].copy()
    selected_time = timestamps.loc[selected.index]
    if selected.empty:
        raise ValueError(f"{path.name} has no data for {year}")

    hour_index = selected_time.dt.floor("h")
    require_four_intervals_per_hour(hour_index, path.name, year)
    interval_kwh = pd.to_numeric(selected["generation(kWh)"], errors="raise")
    hourly = interval_kwh.groupby(hour_index).sum().reindex(expected_hour_index(year))
    hourly.name = "pv_kwh"
    return hourly


def prepare_hourly_temperature(path: Path, year: int) -> pd.Series:
    """Average minute-level temperature hourly and fill missing full hours."""
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    required = {"Time", "Temp (Degree Celsius)"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(raw["Time"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")
    if timestamps.duplicated().any():
        raise ValueError(f"{path.name} contains duplicate timestamps")

    selected = raw.loc[timestamps.dt.year == year, "Temp (Degree Celsius)"]
    selected_time = timestamps.loc[selected.index]
    temperature = pd.Series(
        pd.to_numeric(selected, errors="raise").to_numpy(),
        index=selected_time,
    )
    hourly = temperature.resample("h").mean().reindex(expected_hour_index(year))
    missing_hours = int(hourly.isna().sum())

    weekly_estimate = (hourly.shift(24 * 7) + hourly.shift(-24 * 7)) / 2.0
    hourly = hourly.fillna(weekly_estimate)
    hourly = hourly.interpolate(method="time").ffill().bfill()
    if hourly.isna().any():
        raise ValueError(f"Unable to fill all missing temperatures in {path.name}")

    hourly.name = "outdoor_temperature_c"
    print(f"Prepared outdoor temperature: filled {missing_hours} missing full hours")
    return hourly


def validate_profile(profile: pd.DataFrame, name: str, year: int) -> None:
    """Enforce the output contract used by buildings.py."""
    expected = expected_hour_index(year)
    expected_columns = ["load_kwh", "pv_kwh", "outdoor_temperature_c"]
    if list(profile.columns) != expected_columns:
        raise ValueError(f"{name}: unexpected columns {profile.columns.tolist()}")
    if not profile.index.equals(expected):
        raise ValueError(f"{name}: timestamps are not a complete hourly year")
    if profile.isna().any().any():
        raise ValueError(f"{name}: profile contains missing values")
    if (profile[["load_kwh", "pv_kwh"]] < -1e-9).any().any():
        raise ValueError(f"{name}: profile contains negative energy")
    if len(profile) != len(expected):
        raise ValueError(f"{name}: expected {len(expected)} rows, got {len(profile)}")


def build_annual_profile(
    load_path: Path,
    pv_path: Path,
    hourly_temperature: pd.Series,
    year: int,
    pv_scaling_weight: float = 1.0,
) -> pd.DataFrame:
    if pv_scaling_weight < 0:
        raise ValueError("pv_scaling_weight must be non-negative")

    load = prepare_hourly_load(load_path, year)
    pv = prepare_hourly_pv(pv_path, year) * pv_scaling_weight
    profile = pd.concat([load, pv, hourly_temperature], axis=1)
    profile.index.name = "timestamp"
    profile[["load_kwh", "pv_kwh"]] = profile[["load_kwh", "pv_kwh"]].clip(lower=0.0)
    return profile


def building_configs(year: int) -> dict[str, dict[str, Path]]:
    return {
        "Residential 1": {
            "load_file": DATA_DIR / "6zone_Residential House.csv",
            "pv_file": DATA_DIR / "SQ1.csv",
            "output_file": OUTPUT_DIR / f"residential_1_{year}.csv",
        },
        "Residential 2": {
            "load_file": DATA_DIR / "6zone_Residential House_C_1_new.csv",
            "pv_file": DATA_DIR / "SQ Apartment1-12.csv",
            "output_file": OUTPUT_DIR / f"residential_2_{year}.csv",
        },
        "Large Commercial 1": {
            "load_file": DATA_DIR / "2zone_Supermarket.csv",
            "pv_file": DATA_DIR / "Indoor Sports Centre.csv",
            "output_file": OUTPUT_DIR / f"large_commercial_1_{year}.csv",
        },
        "Large Commercial 2": {
            "load_file": DATA_DIR / "5zone_RetailStandalone.csv",
            "pv_file": DATA_DIR / "Library.csv",
            "output_file": OUTPUT_DIR / f"large_commercial_2_{year}.csv",
        },
    }


def save_typical_day_plot(
    profiles: dict[str, pd.DataFrame],
    plot_date: str,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for (building_name, profile), axis in zip(profiles.items(), axes.flat):
        daily = profile.loc[plot_date]
        hours = daily.index.hour
        axis.plot(hours, daily["load_kwh"], label="Load", linewidth=2)
        axis.plot(hours, daily["pv_kwh"], label="PV", linewidth=2)
        axis.set_title(building_name)
        axis.set_ylabel("Hourly energy (kWh)")
        axis.set_xticks(range(0, 24, 3))
        axis.legend()

    fig.supxlabel("Hour")
    fig.suptitle(f"Hourly load and PV — {plot_date}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved typical-day plot: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare 2023 hourly building load/PV/temperature CSVs."
    )
    parser.add_argument("--year", type=int, default=TARGET_YEAR)
    parser.add_argument("--plot-date", default=PLOT_DATE)
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Do not write the typical-day inspection figure.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_stdio()
    args = parse_args()
    year = args.year
    configs = building_configs(year)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    legacy_output_dir = DATA_DIR / "prepared"
    for config in configs.values():
        (legacy_output_dir / config["output_file"].name).unlink(missing_ok=True)
    for stale_name in ("BG_buildings_2022-06-01.csv", "BG_group_2022-06-01.csv"):
        (legacy_output_dir / stale_name).unlink(missing_ok=True)

    hourly_temperature = prepare_hourly_temperature(TEMPERATURE_FILE, year)
    profiles = {}
    for building_name, config in configs.items():
        profile = build_annual_profile(
            config["load_file"],
            config["pv_file"],
            hourly_temperature,
            year,
            pv_scaling_weight=PV_SCALING_WEIGHTS[building_name],
        )
        validate_profile(profile, building_name, year)
        profile.to_csv(config["output_file"], index=True)
        profiles[building_name] = profile
        print(
            f"Saved {building_name}: {config['output_file']} ({len(profile)} rows, "
            f"PV weight={PV_SCALING_WEIGHTS[building_name]})"
        )

    summary = pd.DataFrame(
        [
            {
                "building": name,
                "output_file": str(configs[name]["output_file"]),
                "pv_scaling_weight": PV_SCALING_WEIGHTS[name],
                "rows": len(profile),
                "start": profile.index.min(),
                "end": profile.index.max(),
                "missing_values": int(profile.isna().sum().sum()),
                "annual_load_kwh": profile["load_kwh"].sum(),
                "annual_pv_kwh": profile["pv_kwh"].sum(),
                "minimum_temperature_c": profile["outdoor_temperature_c"].min(),
                "maximum_temperature_c": profile["outdoor_temperature_c"].max(),
            }
            for name, profile in profiles.items()
        ]
    ).set_index("building")
    expected_rows = len(expected_hour_index(year))
    if not summary["rows"].eq(expected_rows).all():
        raise AssertionError(f"Expected {expected_rows} rows in every profile.")
    if not summary["missing_values"].eq(0).all():
        raise AssertionError("Prepared profiles contain missing values.")
    print("\nProfile summary:")
    print(summary.to_string())

    if not args.skip_plot:
        save_typical_day_plot(
            profiles,
            args.plot_date,
            OUTPUT_DIR / f"building_typical_day_{args.plot_date}.png",
        )


if __name__ == "__main__":
    main()
