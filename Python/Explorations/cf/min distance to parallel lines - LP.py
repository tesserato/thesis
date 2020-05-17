import pulp
import plotly.graph_objects as go
import numpy as np


#######################
n = 7
X = np.arange(n)
f = 1
p = np.pi
W = np.cos(p + 2 * np.pi * f * X / n)
#######################

prob = pulp.LpProblem("LP", pulp.LpMinimize)

x = pulp.LpVariable("p")
y = pulp.LpVariable("f")

objective = []
for t in range(n):
  r = pulp.LpVariable(f"r_{t}")
  k = pulp.LpVariable(f"k_{t}", 0, n, cat=pulp.LpInteger)
  # prob += k == 1
  objective.append(r)
  d = (n*x + 2*np.pi*t*y)/np.sqrt(n**2 + 4*np.pi**2*t**2)
  x_vec = (2 * np.pi * n**2) / (n**2 + 4 * np.pi**2 * t**2)     # p
  y_vec = (4 * np.pi**2 * n * t) / (n**2 + 4 * np.pi**2 * t**2) # f
  m_vec = np.sqrt(x_vec**2 + y_vec**2)
  prob += r == d - m_vec * k

prob += pulp.lpSum([objective[t] for t in range(n)])
prob.writeLP("LP.lp")

status = prob.solve()
# status = prob.solve(pulp.GLPK(msg = 0))

print(pulp.LpStatus[status])

print(f"f={round(pulp.value(y), 2)} p={round(pulp.value(x), 2)}")



