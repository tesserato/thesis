import pulp
import random
import numpy as np

#######################
random.seed(0)
n=10

X = np.arange(n)
W = np.zeros(n)
number_of_random_waves = 1
A = np.array([random.uniform(1, 5) for i in range(number_of_random_waves)])
F = np.array([random.uniform(1, n/2) for i in range(number_of_random_waves)])
P = np.array([random.uniform(0, 2 * np.pi) for i in range(number_of_random_waves)])
W = np.sum(A * np.cos(P + 2 * np.pi * F.T * X[:, np.newaxis] / n), 1)
W = W / np.max(np.abs(W))
idx = np.argmax(A)
# print(W)
print(f"f={round(F[idx], 2)} p={round(P[idx], 2)}")
#######################

#######################
# n = 30
# X = np.arange(n)
# f0 = 3.5
# p0 = 1.4
# W = np.cos(p0 + 2 * np.pi * f0 * X / n)
#######################

prob = pulp.LpProblem("LP", pulp.LpMinimize)

p = pulp.LpVariable("p", 0 , 2 * np.pi)
f = pulp.LpVariable("f", 0 , n / 2)

O = [] # opt

for x in range(n):
  A = []
  for k in range(x+1):    
    a = pulp.LpVariable(f"a_{x}_{k}", 0)
    b = pulp.LpVariable(f"b_{x}_{k}", 0)
    den = np.sqrt(n**2 + 4 * np.pi**2 * x**2)
    if W[x] >= 0:
      d = (n * p + 2 * np.pi * x * f - 4 * np.pi**2 * k * x) / den
    else:
      d = (n * p + 2 * np.pi * x * f - 4 * np.pi**2 * k * x - np.pi * n) / den

    prob += a >= + d # pulp.LpConstraint(a - d, rhs=0 , sense=pulp.LpConstraintGE, name=f"c_{x}_{k}_pos") #
    prob += a >= - d # pulp.LpConstraint(a + d, rhs=0 , sense=pulp.LpConstraintLE, name=f"c_{x}_{k}_neg") #
    
    O.append(a * abs(W[x]))
    A.append(d)
  # prob += pulp.Lpsum([a for a in O]) == pulp. ([a for a in O])
  # prob += pulp.LpConstraint(pulp.lpSum([a for a in O]), rhs=1 ,sense=pulp.LpConstraintGE)


prob += pulp.lpSum([a for a in O]) # <|<|<|<|<|<|<|<|

prob.writeLP("lp.lp")

# status = prob.solve()
# status = prob.solve(pulp.GLPK(msg = 0))
status = prob.solve(pulp.CPLEX_CMD(path=r"C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\bin\x64_win64\cplex.exe"))

print(pulp.LpStatus[status])

for v in prob.variables():
  print(v.name, "=", v.varValue)

print(f"f={round(pulp.value(f), 2)} p={round(pulp.value(p), 2)}")


