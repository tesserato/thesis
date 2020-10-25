import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
from scipy.linalg import block_diag
import numpy.polynomial.polynomial as poly

def constrained_least_squares_arbitrary_intervals(X, Y, I:list, k=3):
  '''constrained least squares with q intervals and k-1 polynomial'''
  assert len(X) == len(Y)
  n = len(X)
  I = [0] + I + [n]

  # U = np.zeros((n, k * q))

  Q = [] # Q

  for i in range(len(I) - 1):
    l0, l1 = I[i], I[i + 1]
    Qi = np.zeros((l1 - l0, k + 1))
    for l in range(l1 - l0):
      for c in range(k + 1):
        Qi[l, c] = X[l0 + l]**c
    Q.append(Qi)

  Q = block_diag(*Q)

  V = np.zeros((2 * (len(I) - 2), (k + 1) + (k + 1) * (len(I) - 2)))

  for i in range(len(I) - 2):
    V[2 * i, i * (k + 1)] = 1
    V[2 * i, (i + 1) * (k + 1)] = -1
    for c in range(1, k + 1):
      V[2 * i, i * (k + 1) + c] = (I[i+1] - .5)**c
      V[2 * i + 1, i * (k + 1) + c] = (I[i+1] - .5)**(c-1)*c
      V[2 * i, (i + 1) * (k + 1) + c] = -(I[i+1] - .5)**c
      V[2 * i + 1, (i + 1) * (k + 1) + c] = -(I[i+1] - .5)**(c-1)*c

  np.savetxt("Q.csv", np.round(Q, 2), delimiter=",")
  np.savetxt("V.csv", np.round(V, 2), delimiter=",")

  '''solving'''
  QTQinv = np.linalg.inv(Q.T @ Q)
  tau = np.linalg.inv(V @ QTQinv @ V.T)
  QTY = Q.T @ Y
  A = QTQinv @ (QTY - V.T @ tau @ V @ QTQinv @ QTY)
  return np.reshape(A, (len(I) - 1, -1))

def coefs_to_array_arbitrary_intervals(A, X, I, n):
  '''evaluates x E X from a coefficient matrix A'''
  k = A.shape[1]
  Y = np.zeros(n)
  Xs = [0] + [int(round(X[i])) for i in I] + [n]

  for i in range(len(Xs) - 1):
    x0, x1 = Xs[i], Xs[i + 1]
    for x in range(x0 , x1):
      for c in range(k):
        Y[x] += A[i, c] * x**c

  # x = len(Y)
  # while x < n:
  #   y = 0
  #   for i in range(k):
  #     y += A[q-1, i] * X[x]**i
  #   Y.append(y)
  #   x += 1
  return Y


'''==============='''
''' Read wav file '''
'''==============='''

name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n, dtype=np.float64)

Xpos, Xneg = se.get_frontiers(W)

posL = []
for i in range(1, Xpos.size):
  posL.append(Xpos[i] - Xpos[i - 1])

negL = []
for i in range(1, Xneg.size):
  negL.append(Xneg[i] - Xneg[i - 1])

if np.std(posL) < np.std(negL):
  print("using positive frontier")
  Xp = Xpos
  maxL = np.max(np.array(posL))
  avgL = int(round(np.average(np.array(posL))))
  pseudoCyclesX = []
  pseudoCyclesY = []
  for i in range(1, Xpos.size):
    x0 = Xpos[i - 1]
    x1 = Xpos[i]
    # a = np.max(np.abs(W[x0 : x1]))
    ft = np.fft.rfft(W[x0 : x1])
    npulse = np.fft.irfft(ft, maxL)
    pseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
    pseudoCyclesX.append(np.arange(maxL))
  pseudoCyclesX = np.array(pseudoCyclesX)
  pseudoCyclesY = np.array(pseudoCyclesY)
else:
  print("using negative frontier")
  maxL = np.max(np.array(negL))
  Xp = Xneg
  avgL = int(round(np.average(np.array(negL))))
  pseudoCyclesX = []
  pseudoCyclesY = []
  for i in range(1, Xpos.size):
    x0 = Xpos[i - 1]
    x1 = Xpos[i]
    # a = np.max(np.abs(W[x0 : x1]))
    ft = np.fft.rfft(W[x0 : x1])
    npulse = np.fft.irfft(ft, maxL)
    pseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
    pseudoCyclesX.append(np.arange(maxL))
  pseudoCyclesX = np.array(pseudoCyclesX)
  pseudoCyclesY = np.array(pseudoCyclesY)

print(f"Max L = {maxL}")

pseudoCyclesY_avg = np.average(pseudoCyclesY, 0)

'''============================================================================'''
'''                                   APROX PC                                 '''
'''============================================================================'''
assert pseudoCyclesY_avg.size == maxL
m = int(maxL)
X01 = np.linspace(0, 1, m, dtype=np.float64)
k = 7
Q = np.zeros((m, k + 1))

for i in range(m):
  for j in range(k + 1):
    Q[i, j] = X01[i]**j

Q = block_diag(Q, Q)

D = np.eye(k + 1)

D = np.hstack((D, -D))

E = np.zeros((2, 2 * (k + 1)))
for j in range(k + 1):
  E[0, j] = X01[0]**j
  E[0, k + j + 1] = -X01[m - 1]**j

for j in range(1, k + 1):
  E[1, j] = j * X01[0] ** (j - 1)
  E[1, k + j + 1] = -j * X01[m - 1] ** (j - 1)

V = np.vstack((D, E))

# np.savetxt("V.csv", V, delimiter=",")

Y = np.hstack((pseudoCyclesY_avg, pseudoCyclesY_avg)).T

print(Q.shape, Y.shape)

QTQinv = np.linalg.inv(Q.T @ Q)
tau = np.linalg.inv(V @ QTQinv @ V.T)
QTY = Q.T @ Y
A = QTQinv @ (QTY - V.T @ tau @ V @ QTQinv @ QTY)
A = np.reshape(A, (2, -1))[0, :]

print(A)

'''============================================================================'''
'''                             MAKE BASE WAVE                                 '''
'''============================================================================'''

Wparam = []
Xparam = []
for x in range(Xp[0]):
  Xparam.append(x)
  Wparam.append(0)
for i in range(1, Xp.size):
  x0 = Xp[i - 1]
  x1 = Xp[i]
  Xl = np.linspace(0, 1, int(x1) - int(x0) + 1, dtype=np.float64)
  Yl = poly.polyval(Xl, A)
  for j in range(Yl.size - 1):
    Xparam.append(x0 + j)
    Wparam.append(Yl[j])
for x in range(Xp[-1], n):
  Xparam.append(x)
  Wparam.append(0)

Wparam = np.array(Wparam) #* amp

print(f"Wn = {W.size}, Wpn = {Wparam.size}")

# print(Wparam)

se.save_wav(Wparam)

'''============================================================================'''
'''                             SPLIT WAVE                                 '''
'''============================================================================'''
n_intervals = 3

limit = np.sum(np.abs(W)) / n_intervals
Ix = []
x0 = 0

for _ in range(n_intervals - 1):
  curr_sum = W[x0]
  while curr_sum < limit:
    x0+= 1
    curr_sum += np.abs(W[x0])
  Ix.append(x0 - 1)

# Ix = np.array(Ix)

print(f"Ix = {Ix}")

'''============================================================================'''
'''                                  CLMS                                      '''
'''============================================================================'''

# Ae = constrained_least_squares_arbitrary_intervals(Wparam, W, Ix, 3)
# We = coefs_to_array_arbitrary_intervals(Ae, Wparam, Wparam[Ix], n)

Ae = constrained_least_squares_arbitrary_intervals(X, np.abs(W), Ix, 3)
We = coefs_to_array_arbitrary_intervals(Ae, X, Ix, n)

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",
  # yaxis = dict(scaleanchor = "x", scaleratio = 1 ),                   # <|<|<|<|<|<|<|<|<|<|<|<|
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

fig.add_trace(
  go.Scatter(
    name="original", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=W,
    # fill="toself",
    mode="lines",
    line=dict(
        width=4,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Approx", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xparam,
    y=Wparam * amp,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="black",
        # dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)


Xintervals = []
Yintervals = []
for x in Ix:
  Xintervals.append(x)
  Xintervals.append(x)
  Xintervals.append(None)
  Yintervals.append(-amp)
  Yintervals.append(amp)
  Yintervals.append(None)

fig.add_trace(
  go.Scatter(
    name="Breaks", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xintervals,
    y=Yintervals,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="silver",
        # dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Env", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xintervals,
    y=We,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="silver",
        # dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))