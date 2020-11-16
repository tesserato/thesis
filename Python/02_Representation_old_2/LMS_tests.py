import numpy as np
import plotly.graph_objects as go

m = 500
k = 3
X = np.arange(m)
Y = np.cos(X/100) + np.random.randn(m)/5

'''Least Squares'''

Q = np.zeros((m, k + 1))

for i in range(m):
  for j in range(k + 1):
    Q[i, j] = i**j

A = np.linalg.inv(Q.T @ Q) @ Q.T @ Y
Y_lms = np.zeros((m))

for i in range(m):
  for j in range(k + 1):
    Y_lms[i] += A[j] * i**j



'''Constrained Least Squares (with q intervals and k-1 polynomial)'''
s = 2
l = m // s
Q = np.zeros((m, k * s))
V = np.zeros((2 * (s-1), k * s))

'''populating Q'''
for i1 in range(s):
  for i2 in range(l * i1, l * (i1 + 1)):
    for i3 in range(k):
      Q[i2, i1 * k + i3] = X[i2]**i3

'''populating V (constraints)'''
for i in range(s - 1):
  for j in range(k):
    V[2 * i, i * k + j] = X[(i + 1) * l]**j
    V[2 * i, i * k + k + j] = -X[(i + 1) * l]**j
for i in range(s - 1):
  for j in range(1, k):
    V[2 * i + 1, i * k + j] = j * X[(i + 1) * l] ** (j - 1)
    V[2 * i + 1, i * k + k + j] = -j * X[(i + 1) * l] ** (j - 1)

'''solving'''
# UTUinv = np.linalg.inv(np.matmul(Q.T, Q))
# tau = np.linalg.inv(np.matmul(np.matmul(V, UTUinv),V.T))
# UTW = np.matmul(Q.T, Y)
# par1 = np.matmul(np.matmul(V, UTUinv), UTW)
# A = np.matmul(UTUinv,UTW - np.matmul(np.matmul(V.T, tau),par1))
# A = np.reshape(A, (s, -1))

QTQinv = np.linalg.inv(Q.T @ Q)
tau = np.linalg.inv(V @ QTQinv @ V.T)
QTY = Q.T @ Y
# par1 = V @ QTQinv @ QTY
A = QTQinv @ (QTY - V.T @ tau @ V @ QTQinv @ QTY)
A = np.reshape(A, (s, -1))

'''evaluates y E Y from a coefficient matrix A'''
Y_clms = []
for i in range(s):
  for x in range(i * l, (i + 1) * l):
    y = 0
    for j in range(k):
      y += A[i, j] * X[x]**j
    Y_clms.append(y)
  # Y_clms.append(None)

x = len(Y_clms)
while x < m:
  y = 0
  for i in range(k):
    y += A[s-1, i] * X[x]**i
  Y_clms.append(y)
  x += 1

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white"

fig.add_trace(
  go.Scatter(
    name="Data", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
    mode="markers",
    # line=dict(
    #     # size=8,
    #     color="silver",
    #     # showscale=False
    # ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="lms", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y_lms,
    mode="markers+lines",
    # mode="lines",
    # line=dict(
    #     # size=8,
    #     color="silver",
    #     # showscale=False
    # ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="clms", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=Y_clms,
    mode="markers+lines",
    # mode="lines",
    # line=dict(
    #     # size=8,
    #     color="silver",
    #     # showscale=False
    # ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))