import pandapower as pp
import pandapower.networks as pn

# print([name for name in dir(pn)])
net = pn.create_cigre_network_lv()

# print(net.bus)
# print(net.line)
print(net.load["p_mw"].sum())
# print(net.trafo)
# print(net.ext_grid)
# print(net.switch)
# print(net.sgen)
print(net.storage)

# def base_solver(net: Optional[pp.pandapowerNet] = None, T: int = 24) -> Dict[str, Any]:
#     if net is None:
#         net = pn.case33bw()

#     # extract constants
#     constants = extract_constants(input=input, net=net, T=T)
#     lambda_g = constants["lambda_g"]
#     pi_min = constants["pi_min"]
#     pi_max = constants["pi_max"]
#     p0_max = constants["p0_max"]
#     q0_max = constants["q0_max"]
#     v_ref = constants["v_ref"]
#     v_min = constants["v_min"]
#     v_max = constants["v_max"]
#     load_curve = constants["load_curve"]
#     mg_buses = [int(b) for b in constants["mg_buses"]]
#     n_bus = constants["n_bus"]
#     n_line = constants["n_line"]
#     slack = constants["slack"]
#     from_bus = constants["from_bus"]
#     to_bus = constants["to_bus"]
#     r = constants["r"]
#     x = constants["x"]
#     l_max = constants["l_max"]
#     s_base_mva = constants["s_base_mva"]
#     verbose = constants["verbose"]

#     # build children and parent_edge
#     children = {i: [] for i in range(n_bus)}
#     parent_edge = {i: None for i in range(n_bus)}
#     for e in range(n_line):
#         children[from_bus[e]].append(e)
#         parent_edge[to_bus[e]] = e

#     passive = _build_passive_profiles(net, T, load_curve, s_base_mva=s_base_mva)
#     p_passive = passive["p_passive"]
#     q_passive = passive["q_passive"]

#     model = gp.Model("leader_lin_solver")
#     model.Params.OutputFlag = int(bool(verbose))

#     pi = model.addVars(T, lb=pi_min, ub=pi_max, name="pi")
#     p0 = model.addVars(T, lb=0.0, ub=p0_max, name="p0")
#     q0 = model.addVars(T, lb=0.0, ub=q0_max, name="q0")

#     v = model.addVars(n_bus, T, lb=v_min ** 2, ub=v_max ** 2, name="v")
#     l = model.addVars(n_line, T, lb=0.0, name="l")
#     p = model.addVars(n_line, T, lb=-gp.GRB.INFINITY, name="p")
#     q = model.addVars(n_line, T, lb=-gp.GRB.INFINITY, name="q")

#     p_inj = model.addVars(n_bus, T, lb=-gp.GRB.INFINITY, name="p_inj")
#     q_inj = model.addVars(n_bus, T, lb=-gp.GRB.INFINITY, name="q_inj")
#     p_mg = model.addVars(mg_buses, T, lb=-gp.GRB.INFINITY, name="p_mg")
#     q_mg = model.addVars(mg_buses, T, lb=-gp.GRB.INFINITY, name="q_mg")

#     model.setObjective(
#         gp.quicksum(lambda_g[t] * p0[t] for t in range(T))
#         - gp.quicksum(pi[t] * p_mg[m, t] for m in mg_buses for t in range(T)),
#         gp.GRB.MINIMIZE,
#     )

#     for t in range(T):
#         model.addConstr(v[slack, t] == v_ref ** 2, name=f"slack_v[{t}]")

#         for e in range(n_line):
#             i = from_bus[e]
#             j = to_bus[e]
#             model.addConstr(l[e, t] <= l_max[e], name=f"l_bound[{e},{t}]")
#             model.addConstr(
#                 v[j, t] == v[i, t] - 2.0 * (r[e] * p[e, t] + x[e] * q[e, t]) + (r[e] ** 2 + x[e] ** 2) * l[e, t],
#                 name=f"voltage_drop[{e},{t}]",
#             )
#             model.addQConstr(
#                 p[e, t] * p[e, t] + q[e, t] * q[e, t] <= l[e, t] * v[i, t],
#                 name=f"soc[{e},{t}]",
#             )

#         for j in range(n_bus):
#             if j in mg_buses:
#                 model.addConstr(p_inj[j, t] == p_mg[j, t] + p_passive[j, t], name=f"p_inj_def[{j},{t}]")
#                 model.addConstr(q_inj[j, t] == q_mg[j, t] + q_passive[j, t], name=f"q_inj_def[{j},{t}]")
#             else:
#                 model.addConstr(p_inj[j, t] == p_passive[j, t], name=f"p_inj_passive[{j},{t}]")
#                 model.addConstr(q_inj[j, t] == q_passive[j, t], name=f"q_inj_passive[{j},{t}]")

#             pe = parent_edge[j]
#             if pe is None:
#                 continue

#             outgoing_p = gp.quicksum(p[e, t] for e in children[j])
#             outgoing_q = gp.quicksum(q[e, t] for e in children[j])
#             model.addConstr(outgoing_p == p[pe, t] - p_inj[j, t] - r[pe] * l[pe, t], name=f"p_balance[{j},{t}]")
#             model.addConstr(outgoing_q == q[pe, t] - q_inj[j, t] - x[pe] * l[pe, t], name=f"q_balance[{j},{t}]")

#         model.addConstr(
#             p0[t] == gp.quicksum(p[e, t] for e in children[slack]) + p_inj[slack, t],
#             name=f"slack_p_balance[{t}]",
#         )
#         model.addConstr(
#             q0[t] == gp.quicksum(q[e, t] for e in children[slack]) + q_inj[slack, t],
#             name=f"slack_q_balance[{t}]",
#         )

#     model.optimize()
#     if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
#         raise RuntimeError(f"Gurobi failed with status code {model.Status}.")

#     return {
#         "status": int(model.Status),
#         "objective": float(model.ObjVal),
#         "pi": np.array([pi[t].X for t in range(T)]),
#         "p0": np.array([p0[t].X for t in range(T)]),
#         "q0": np.array([q0[t].X for t in range(T)]),
#         "p_mg": {m: np.array([p_mg[m, t].X for t in range(T)]) for m in mg_buses},
#         "q_mg": {m: np.array([q_mg[m, t].X for t in range(T)]) for m in mg_buses},
#         "load_curve": load_curve,
#     }
