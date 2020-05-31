import pulp
# import plotly.graph_objects as go
import numpy as np


#######################
n = 100
X = np.arange(n)
f = 3.5
p = 1.4
W = np.cos(p + 2 * np.pi * f * X / n)
#######################

prob = pulp.LpProblem("LP", pulp.LpMinimize)

x = pulp.LpVariable("p", 0 , 2 * np.pi)
y = pulp.LpVariable("f", 0 , n/2)

O = [] # opt
K = []
for t in range(n):
  m = 2 * np.pi * n / np.sqrt(n**2 + 4 * np.pi**2 * t**2)
  d = (n*x + 2*np.pi*t*y)/np.sqrt(n**2 + 4*np.pi**2 * t**2)
  k = pulp.LpVariable(f"k_{t}", 0, t + 1, cat=pulp.LpInteger)
  a = pulp.LpVariable(f"a_{t}", 0, t + 1)
  if W[t] >= 0:
    r = d - k * m
  else:
    r = d - k * m + m/2
  prob += a >= r
  prob += a >= - r
  O.append(a)

prob += pulp.lpSum([O[t] * abs(W[t]) for t in range(n)]) # <|<|<|<|<|<|<|<|


prob.writeLP("lp.lp")

# status = prob.solve()
# status = prob.solve(pulp.GLPK(msg = 0))
status = prob.solve(pulp.CPLEX())

print(pulp.LpStatus[status])

for v in prob.variables():
  print(v.name, "=", v.varValue)

print(f"f={round(pulp.value(y), 2)} p={round(pulp.value(x), 2)}")


