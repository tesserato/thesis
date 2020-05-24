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

x = pulp.LpVariable("p")
y = pulp.LpVariable("f")

O = [] # opt
K = []
for t in range(n):
  d = (n*x + 2*np.pi*t*y)/np.sqrt(n**2 + 4*np.pi**2 * t**2)

  x_vec = (2 * np.pi * n**2) / (n**2 + 4 * np.pi**2 * t**2)     # p
  y_vec = (4 * np.pi**2 * n * t) / (n**2 + 4 * np.pi**2 * t**2) # f
  m_vec = np.sqrt(x_vec**2 + y_vec**2)

  k = pulp.LpVariable(f"k_{t}", cat=pulp.LpInteger)
  K.append(k)
  # prob += k <= (d / m_vec) + 0.5
  # prob += k >= (d / m_vec) - 0.5
  r = d - m_vec * k                                 # <|<|<|<|<|<|<|<|
  a = pulp.LpVariable(f"a_{t}")
  if W[t] >= 0: # residue abs value must be the smallest possible
    prob += a >= r
    prob += a >= -r
  else:       # W[t] < 0 -> residue abs value must be the highest possible
    prob += a <= r
    prob += a <= -r
  O.append(a)

prob += pulp.lpSum([O[t] * (W[t]) for t in range(n)]) # <|<|<|<|<|<|<|<|


prob.writeLP("LP.lp")

# status = prob.solve()
status = prob.solve(pulp.GLPK(msg = 0))

print(pulp.LpStatus[status])

print(f"f={round(pulp.value(y), 2)} p={round(pulp.value(x), 2)}")

for t, a in enumerate(O):
  print(f"a_{t}={pulp.value(a)}")

for t, k in enumerate(K):
  print(f"k_{t}={pulp.value(k)}")

