import numpy as np
from scipy.linalg import block_diag
import numpy.polynomial.polynomial as poly

def split_raw_frontier(Xf, W):
  Areas = []
  for i in range(1, Xf.size):
    x0 = Xf[i - 1]
    x1 = Xf[i]
    y0 = np.abs(W[x0])
    y1 = np.abs(W[x1])
    a = (x1 - x0) * (y0 + y1) / 2
    Areas.append(a)

  Areas = np.array(Areas)

  n_intervals = 4

  limit = np.sum(Areas) / n_intervals

  Ix = []
  i = 0
  for _ in range(n_intervals - 1):
    curr_sum = Areas[i]
    while curr_sum < limit:
      i += 1
      curr_sum += Areas[i]
    dx = ()
    Ix.append(i)
  return Ix

def constrained_least_squares_arbitrary_intervals(X, W, I:list, k=3):
  '''constrained least squares with intervals as defined by the x coordinates in I and k polynomial'''
  Y = W[X]
  assert len(X) == len(Y)
  # n = len(X)
  I = [0] + I + [X.size - 1]

  Qlist = []

  for i in range(len(I) - 1):
    l0, l1 = I[i], I[i + 1]
    Qi = np.zeros((l1 - l0 + 1, k + 1))
    for l in range(l1 - l0 + 1):
      for c in range(k + 1):
        Qi[l, c] = X[l0 + l]**c
    Qlist.append(Qi)

  Q = np.zeros((len(X), (len(I) - 1) * (k + 1)))

  l0 = 0
  c0 = 0
  for q in Qlist:
    print(q.shape)
    l1 = l0 + q.shape[0]
    c1 = c0 + q.shape[1]
    Q[l0:l1, c0:c1] = q
    l0 = l1 - 1
    c0 = c1

  # Q = block_diag(*Q)

  V = np.zeros((2 * (len(I) - 2), (k + 1) + (k + 1) * (len(I) - 2)))

  # Xf[x0 + 1] - Xf[x0]) * np.abs(W[Xf[x0 + 1]]) / (np.abs(W[Xf[x0]]) + np.abs(W[Xf[x0 + 1]])

  for i in range(len(I) - 2):
    V[2 * i, i * (k + 1)] = 1
    V[2 * i, (i + 1) * (k + 1)] = -1
    x = X[I[i + 1]] + 0.5
    # x0 = X[I[i + 1]]
    # x1 = X[I[i + 2]]
    # x = (x1 - x0) * np.abs(W[x1]) / (np.abs(W[x0]) + np.abs(W[x1]))
    for c in range(1, k + 1):
      V[2 * i, i * (k + 1) + c] = x**c
      V[2 * i + 1, i * (k + 1) + c] = x**(c-1) * c
      V[2 * i, (i + 1) * (k + 1) + c] = -x**c
      V[2 * i + 1, (i + 1) * (k + 1) + c] = -x**(c-1) * c

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

  x = len(Y)
  # while x < n: # TODO
  #   y = 0
  #   for i in range(k):
  #     y += A[q-1, i] * X[x]**i
  #   Y.append(y)
  #   x += 1
  return Y

def pseudocycles_average(Xpos, Xneg, W):
  posL = []
  for i in range(1, Xpos.size):
    posL.append(Xpos[i] - Xpos[i - 1])

  negL = []
  for i in range(1, Xneg.size):
    negL.append(Xneg[i] - Xneg[i - 1])

  if np.std(posL) < np.std(negL):
    print("using positive frontier")
    used_positive_frontier = True
    maxL = np.max(np.array(posL))
    # avgL = int(round(np.average(np.array(posL))))
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
    used_positive_frontier = False
    # avgL = int(round(np.average(np.array(negL))))
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
  return pseudoCyclesY_avg, used_positive_frontier

def approximate_pseudocycles_average(pseudoCyclesY_avg):
  m = pseudoCyclesY_avg.size
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
  return A

def parametric_W(Xp, A, n):
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
  return Wparam
