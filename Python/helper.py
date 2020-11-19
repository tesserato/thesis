import numpy as np
import numpy.polynomial.polynomial as poly
from statistics import mode
from scipy.linalg import block_diag
from scipy import interpolate
from scipy.signal import savgol_filter

# print(np.finfo(np.float64))
# print(np.finfo(np.double))
# print(np.finfo(np.longdouble))
# exit()

# print(np.iinfo(np.long))
# exit()

def _refine_frontier(Xp, W, avgL, stdL, n_stds = 3):
  "find additional frontier points, and return an array with then, if found"
  if W[Xp[0]] >= 0:
    e = np.argmax
  else:
    e = np.argmin

  print("std=", stdL)
  Pulses = []
  Pulses_to_split = []
  for i in range(1, Xp.size):
    x0 = int(Xp[i - 1])
    x1 = int(Xp[i])
    if x1 - x0 >= avgL + n_stds * stdL:
      Pulses_to_split.append((x0, x1))
    # else:
    #   Pulses_to_split.append((x0, x1))

  while len(Pulses_to_split) > 0:
    Pulses_to_test = []
    for x0, x1 in Pulses_to_split:
      Xzeroes = []
      currsign = np.sign(W[x0])
      for i in range(x0 + 1, x1):
        if currsign != np.sign(W[i]):
          Xzeroes.append(i)
          currsign = np.sign(W[i])
      if len(Xzeroes) > 1:
        x = Xzeroes[0] + e(W[Xzeroes[0] : Xzeroes[-1]])
        Pulses_to_test.append((x0, x))
        Pulses_to_test.append((x, x1))
      # else:
      #   Pulses.append((x0, x1))
    Pulses_to_split = []
    for x0, x1 in Pulses_to_test:
      if x1 - x0 >= avgL + n_stds * stdL:
        Pulses_to_split.append((x0, x1))
      else:
        Pulses.append((x0, x1))

  if len(Pulses) > 0:
    return np.unique(np.array([p[0] for p in Pulses] + [p[1] for p in Pulses], dtype=np.int))
  else:
    print("No refinement found")
    return None

def refine_frontier_iter(Xp, W): # mode
  Xp = Xp.astype(np.int)
  L = Xp[1:] - Xp[:-1]
  avgL = mode(L)
  stdL = np.average(np.abs(L - avgL))
  Xp_new = _refine_frontier(Xp, W, avgL, stdL, 2)

  while not Xp_new is None:
    Xp_c = np.unique(np.hstack([Xp, Xp_new])).astype(np.int)
    L = Xp_c[1:] - Xp_c[:-1]
    avgL = mode(L)
    stdL_c = np.average(np.abs(L - avgL))
    if stdL_c <= stdL:
      # avgL = np.average(L)
      stdL = stdL_c
      Xp = Xp_c
      Xp_new = _refine_frontier(Xp, W, avgL, stdL, 2)
    else:
      print(f"std0={stdL}, std1={stdL_c}")
      break
  return Xp

def constrained_least_squares_arbitrary_intervals(X, Y, I:list, k=2, solve_method="k"):
  '''constrained least squares with intervals as defined by the x coordinates in I and k polynomial'''
  Y = Y.astype(np.float64)
  assert len(X) == len(Y)
  I = [0] + I + [X.size]

  # print(I)

  Qlist = []
  for i in np.arange(len(I) - 1):
    l0, l1 = I[i], I[i + 1]
    Qi = np.zeros((l1 - l0, k + 1), dtype=np.float64)
    for l in np.arange(l1 - l0): # <|<|<|
      for c in np.arange(k + 1):
        Qi[l, c] = X[l0 + l]**c
    Qlist.append(Qi)
  
  Q = (block_diag(*Qlist)).astype(np.float64)

  V = np.zeros((2 * (len(I) - 2), (k + 1) + (k + 1) * (len(I) - 2)), dtype=np.float64)
  for i in np.arange(len(I) - 2):
    V[2 * i, i * (k + 1)] = 1
    V[2 * i, (i + 1) * (k + 1)] = -1
    x = X[I[i + 1] - 1] # + 0.5 #TODO
    for c in np.arange(1, k + 1):
      V[2 * i, i * (k + 1) + c] = x**c
      V[2 * i + 1, i * (k + 1) + c] = x**(c-1) * c
      V[2 * i, (i + 1) * (k + 1) + c] = -(x**c)
      V[2 * i + 1, (i + 1) * (k + 1) + c] = -(x**(c-1) * c)

  # np.savetxt("Q.csv", np.round(Q, 2), delimiter=",")
  # np.savetxt("V.csv", np.round(V, 2), delimiter=",")
  
  if solve_method=="k":
    '''solving kkt'''
    s = len(I) - 1
    a1 = np.hstack([ 2 * Q.T @ Q , V.T                              ])
    a2 = np.hstack([ V           , np.zeros((2 * (s-1), 2 * (s-1))) ])
    x = np.vstack([a1, a2])
    # print((2 * Q.T @ Y).shape)
    y = np.hstack([2 * Q.T @ Y, np.zeros(2 * (s-1))]).T
    A = np.linalg.solve(x, y)[ :s * (k+1)]

  if solve_method=="d":
    '''solving direct'''
    QTQinv = np.linalg.inv(Q.T @ Q)
    tau = np.linalg.inv(V @ QTQinv @ V.T)
    QTY = Q.T @ Y
    A = QTQinv @ (QTY - V.T @ tau @ V @ QTQinv @ QTY)

  return np.reshape(A, (len(I) - 1, -1)).astype(np.float64)

def coefs_to_array_arbitrary_intervals(A, X, I:list, n):
  '''evaluates x E X from a coefficient matrix A'''
  Y = np.zeros(n)
  # print(A.shape)
  I = [0] + I + [n]
  for i in np.arange(A.shape[0]):
    for l in np.arange(I[i], I[i + 1]):
      for c in range(A.shape[1]):
        Y[l] += A[i, c] * X[l]**c
  return Y

def coefs_to_array_arbitrary_intervals_dYdX(A, X, I, n):
  Y = np.zeros(n)
  # print(A.shape)
  I = [0] + I + [n]
  for i in np.arange(A.shape[0]):
    for l in np.arange(I[i], I[i + 1]):
      for c in range(1, A.shape[1]):
        Y[l] += c * A[i, c] * X[l]**(c-1)
  return Y

def average_pc_waveform(Xp, W):
  # amp = np.max(np.abs(W))
  max_T = np.max(np.abs(Xp[1:] - Xp[:-1]))
  Xlocal = np.linspace(0, 1, max_T)

  orig_pcs = []
  norm_pcs = []
  for i in range(1, Xp.size):
    x0 = Xp[i - 1]
    x1 = Xp[i] + 1
    orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      # Ylocal = Ylocal / np.max(np.abs(Ylocal)) * amp
      norm_pcs.append(Ylocal)  
  return np.average(np.array(norm_pcs), 0), orig_pcs, norm_pcs

def approximate_pc_waveform(X, Y, I = [], k=2, solve_method="k"):
  '''constrained least squares with intervals as defined by the x coordinates in I and k polynomial'''
  Y = Y.astype(np.float64)
  assert len(X) == len(Y)
  I = [0] + I + [X.size]

  Qlist = []
  for i in np.arange(len(I) - 1):
    l0, l1 = I[i], I[i + 1]
    Qi = np.zeros((l1 - l0, k + 1), dtype=np.float64)
    for l in np.arange(l1 - l0): # <|<|<|
      for c in np.arange(k + 1):
        Qi[l, c] = X[l0 + l]**c
    Qlist.append(Qi)
  
  Q = (block_diag(*Qlist)).astype(np.float64)

  V = np.zeros((2 * (len(I) - 2) + 2, (k + 1) + (k + 1) * (len(I) - 2)), dtype=np.float64)
  for i in np.arange(len(I) - 2):
    V[2 * i, i * (k + 1)] = 1
    V[2 * i, (i + 1) * (k + 1)] = -1
    x = X[I[i + 1] - 1] + 0.5 #TODO
    for c in np.arange(1, k + 1):
      V[2 * i, i * (k + 1) + c] = x**c
      V[2 * i + 1, i * (k + 1) + c] = x**(c-1) * c
      V[2 * i, (i + 1) * (k + 1) + c] = -(x**c)
      V[2 * i + 1, (i + 1) * (k + 1) + c] = -(x**(c-1) * c)

  V[-2, 0] = 1
  V[-2, -k-1] = -1
  for c in np.arange(1, k + 1):
    V[-2, c] = X[0]**c
    V[-1, c] = c * X[0]**(c-1)
    V[-2, -k - 1 + c] = -(X[-1]**c)
    V[-1, -k - 1 + c] = -(c * X[-1]**(c-1))

  # np.savetxt("Q.csv", np.round(Q, 2), delimiter=",")
  # np.savetxt("V.csv", np.round(V, 2), delimiter=",")
  
  if solve_method=="k":
    '''solving kkt'''
    s = len(I) - 1
    a1 = np.hstack([ 2 * Q.T @ Q , V.T                              ])
    a2 = np.hstack([ V           , np.zeros((2 + 2 * (s-1), 2 + 2 * (s-1))) ])
    x = np.vstack([a1, a2])
    # print((2 * Q.T @ Y).shape)
    y = np.hstack([2 * Q.T @ Y, np.zeros(2 + 2 * (s-1))]).T
    A = np.linalg.solve(x, y)[ :s * (k+1)]

  if solve_method=="d":
    '''solving direct'''
    QTQinv = np.linalg.inv(Q.T @ Q)
    tau = np.linalg.inv(V @ QTQinv @ V.T)
    QTY = Q.T @ Y
    A = QTQinv @ (QTY - V.T @ tau @ V @ QTQinv @ QTY)

  return np.reshape(A, (len(I) - 1, -1)).astype(np.float64)

def constrained_least_squares_arbitrary_intervals_X_to_Y(X, dX, Y, I:list, k=2, solve_method="k"):
  '''constrained least squares with intervals as defined by the x coordinates in I and k polynomial'''
  assert len(X) == len(Y)  
  # X  = X.astype(np.float64)
  # dX = dX.astype(np.float64)
  # Y  = Y.astype(np.float64)
  T  = np.arange(X.size, dtype=np.float64)

  
  I = [0] + I + [X.size - 1]
  Qlist = []
  for i in range(len(I) - 1):
    l0, l1 = I[i], I[i + 1]
    Qi = np.zeros((l1 - l0 + 1, k + 1))
    for l in range(l1 - l0 + 1):
      for c in range(k + 1):
        Qi[l, c] = X[l0 + l] * T[l0 + l]**c
    Qlist.append(Qi)

  Q = np.zeros((len(X), (len(I) - 1) * (k + 1)))
  l0 = 0
  c0 = 0
  for q in Qlist:
    # print(q.shape)
    l1 = l0 + q.shape[0]
    c1 = c0 + q.shape[1]
    Q[l0:l1, c0:c1] = q
    l0 = l1 - 1
    c0 = c1

  V = np.zeros((2 * (len(I) - 2), (k + 1) + (k + 1) * (len(I) - 2)))
  for i in range(len(I) - 2):
    # r = np.random.rand()
    x = X[I[i + 1]]
    dx = dX[I[i + 1]]
    t = T[I[i + 1]]
    V[2 * i, i * (k + 1)] = x
    V[2 * i, (i + 1) * (k + 1)] = -x
    V[2 * i + 1, i * (k + 1)] = dx
    V[2 * i + 1, (i + 1) * (k + 1)] = -dx
    for c in np.arange(1, k + 1):
      # r1 = np.random.rand()
      # r2 = np.random.rand()
      # r3 = np.random.rand()
      # r4 = np.random.rand()
      v1 = x * t**c
      v2 = dx * t**c + c * x * t**(c-1)
      V[2 * i, i * (k + 1) + c] = v1
      V[2 * i + 1, i * (k + 1) + c] = v2
      V[2 * i, (i + 1) * (k + 1) + c] = -v1
      V[2 * i + 1, (i + 1) * (k + 1) + c] = -v2

  # np.savetxt("Q.csv", np.round(Q, 2), delimiter=",")
  # np.savetxt("V.csv", np.round(V, 2), delimiter=",")

  if solve_method=="k":
    '''solving kkt'''
    s = len(I) - 1
    a1 = np.hstack([ 2 * Q.T @ Q , V.T                              ])
    a2 = np.hstack([ V           , np.zeros((2 * (s-1), 2 * (s-1))) ])
    x = np.vstack([a1, a2])
    # print((2 * Q.T @ Y).shape)
    y = np.hstack([2 * Q.T @ Y, np.zeros(2 * (s-1))]).T
    A = np.linalg.solve(x, y)[ :s * (k + 1)]

  if solve_method=="d":
    '''solving direct'''
    QTQinv = np.linalg.inv(Q.T @ Q)
    tau = np.linalg.inv(V @ QTQinv @ V.T)
    QTY = Q.T @ Y
    A = QTQinv @ (QTY - V.T @ tau @ V @ QTQinv @ QTY)

  return np.reshape(A, (len(I) - 1, -1))





def split_raw_frontier(Xf, W, n_intervals = 4):
  Areas = []
  for i in range(1, Xf.size):
    x0 = Xf[i - 1]
    x1 = Xf[i]
    y0 = np.abs(W[x0])
    y1 = np.abs(W[x1])
    a = (x1 - x0) * (y0 + y1) / 2
    Areas.append(a)
  Areas = np.array(Areas)

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



def smooth_savgol(V):
  window_len = int(2 * ((V.size / 20) // 2) + 1)
  return savgol_filter(V, window_len, 3)

def find_zeroes(V):
  '''remove best fitting line from V'''
  X = np.arange(V.size)
  b, a = poly.polyfit(X, V, 1)
  V = V - a * X + b
  '''Smooth V'''
  V = smooth_savgol(V)
  '''find zeroes'''
  s = np.sign(V[0])
  zeroes = []
  for i, v in enumerate(V):
    if np.sign(v) != s or v == 0:
      s = np.sign(-1 * s)
      zeroes.append(i)
  # print(zeroes)
  return zeroes

def find_zeroes_alt(V):
  FT = np.fft.rfft(V)
  f = np.argmax(np.abs(FT))
  print(f"f={f}")
  '''remove best fitting line from V'''
  X = np.arange(V.size)
  A = poly.polyfit(X, V, 1)
  V = V - poly.polyval(X, A)
  '''Smooth V'''
  V = smooth_savgol(V)
  '''find zeroes'''
  s = np.sign(V[0])
  zeroes = []
  for i, v in enumerate(V):
    if np.sign(v) != s or v == 0:
      s = np.sign(-1 * s)
      zeroes.append(i)
  return zeroes

def find_breakpoints(V):
  '''remove best fitting line from V'''
  X = np.arange(V.size)
  A = poly.polyfit(X, V, 1)
  V = V - poly.polyval(X, A)
  '''Smooth V'''
  V = savgol_filter(V, 11, 3)
  '''find zeroes'''
  s = np.sign(V[0])
  zeroes = []
  for i, v in enumerate(V):
    if np.sign(v) != s or v == 0:
      s = np.sign(-1 * s)
      zeroes.append(i)
  return zeroes, V

