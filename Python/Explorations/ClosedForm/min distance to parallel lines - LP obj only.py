import pulp
import plotly.graph_objects as go
import numpy as np


#######################
n = 5
X = np.arange(n)
f = 1
p = np.pi
W = np.cos(p + 2 * np.pi * f * X / n)
#######################

prob = pulp.LpProblem("LP", pulp.LpMinimize)

x = pulp.LpVariable("p", lowBound=0, upBound=2 * np.pi)
y = pulp.LpVariable("f", lowBound=0, upBound=n)

O = [] # opt
K = []
for t in range(n):
  d = (n*x + 2*np.pi*t*y)/np.sqrt(n**2 + 4*np.pi**2 * t**2)

  x_vec = (2 * np.pi * n**2) / (n**2 + 4 * np.pi**2 * t**2)     # p
  y_vec = (4 * np.pi**2 * n * t) / (n**2 + 4 * np.pi**2 * t**2) # f
  m_vec = np.sqrt(x_vec**2 + y_vec**2)

  k = pulp.LpVariable(f"k_{t}", 0, cat=pulp.LpInteger)
  K.append(k)

  r = d - m_vec * k                                 # <|<|<|<|<|<|<|<|

  O.append(r)

prob += pulp.lpSum([O[t] for t in range(n)]) # <|<|<|<|<|<|<|<|


prob.writeLP("LP.lp")

status = prob.solve()
# status = prob.solve(pulp.GLPK(msg = 0))

print(pulp.LpStatus[status])

print(f"f={round(pulp.value(y), 2)} p={round(pulp.value(x), 2)}")

for t, a in enumerate(O):
  print(f"a_{t}={pulp.value(a)}")

for t, k in enumerate(K):
  print(f"k_{t}={pulp.value(k)}")

