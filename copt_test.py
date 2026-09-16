"""A small mixed-integer linear program to test COPT."""

import coptpy as cp
from coptpy import COPT

# max  5x + 3y + 2z
# s.t. 2x +  y + z <= 8
#       x + 3y     <= 6
#      x integer, y binary, z continuous
#      x, y, z >= 0

env = cp.Envr()
model = env.createModel("simple_mip")

x = model.addVar(lb=0.0, ub=5.0, vtype=COPT.INTEGER, name="x")
y = model.addVar(vtype=COPT.BINARY, name="y")
z = model.addVar(lb=0.0, ub=10.0, vtype=COPT.CONTINUOUS, name="z")

model.setObjective(5 * x + 3 * y + 2 * z, sense=COPT.MAXIMIZE)
model.addConstr(2 * x + y + z <= 8, name="c1")
model.addConstr(x + 3 * y <= 6, name="c2")

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Status: OPTIMAL")
    print(f"x = {x.x:.4f}  (integer)")
    print(f"y = {y.x:.4f}  (binary)")
    print(f"z = {z.x:.4f}  (continuous)")
    print(f"obj = {model.objval:.4f}")
else:
    print(f"Status: {model.status}")
