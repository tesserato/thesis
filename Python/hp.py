import numpy as np
from scipy.linalg import block_diag
import numpy.polynomial.polynomial as poly
from collections import Counter
from math import gcd
from statistics import mode


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

  # np.savetxt("Q.csv", np.round(Q, 2), delimiter=",")
  # np.savetxt("V.csv", np.round(V, 2), delimiter=",")

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
  return pseudoCyclesY_avg, used_positive_frontier, pseudoCyclesY

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

def parametric_W(Xp, A, n, approximate=False):
  Wparam = []
  # Xparam = []
  if approximate:
    XX = np.arange(Xp.size)
    C = poly.polyfit(XX, Xp, 3)
    Xp = np.round(poly.polyval(XX, C)).astype(int)
  for x in range(Xp[0]):
    # Xparam.append(x)
    Wparam.append(0)
  for i in range(1, Xp.size):
    x0 = Xp[i - 1]
    x1 = Xp[i]
    Xl = np.linspace(0, 1, int(x1) - int(x0) + 1, dtype=np.float64)
    Yl = poly.polyval(Xl, A)
    for j in range(Yl.size - 1):
      # Xparam.append(x0 + j)
      Wparam.append(Yl[j])
  for x in range(Xp[-1], n):
    # Xparam.append(x)
    Wparam.append(0)

  Wparam = np.array(Wparam) #* amp
  return Wparam[0:n]

def linearize_pc(Xp):
  Xp = Xp.astype(np.int)
  AB = []
  for i in range(Xp.size):
    for j in range(i+1, Xp.size):
      anum = Xp[j] - Xp[i]
      aden = j - i
      agcd = gcd(anum, aden)
      bnum = i * Xp[j] - j * Xp[i]
      bden = i - j
      bgcd = gcd(bnum, bden)
      AB.append((anum / agcd, aden / agcd, bnum / bgcd, bden / bgcd))

  # countsAB = Counter([(ab[0], ab[1]) for ab in AB])

  countsAB = Counter(AB)

  afrac = countsAB.most_common(1)[0][0]

  # B = []
  # for ab in AB:
  #   if afrac == (ab[0], ab[1]):
  #     B.append(ab[2] / ab[3])

  print(f"a num, a den = {countsAB.most_common(10)[0][0]}, total={len(AB)}")

  a = afrac[0] / afrac[1]
  b = afrac[2] / afrac[3]

  x0 = int(np.round(- b / a))
  x1 = int(np.round((Xp[-1]- b) / a))

  X = np.arange(x0, x1)
  return X, a * X + b

def std(V1, V2):
  return np.sqrt(np.average((V1 - V2)**2))

# def refine_frontier(Xp, W):
#   "find additional frontier points, and return an array with then, if found"
#   if W[Xp[0]] >= 0:
#     e = np.argmax
#   else:
#     e = np.argmin
#   L = Xp[1:] - Xp[:-1]
#   avgL = np.average(L)
#   stdL = np.std(L)
#   Xnew = []
#   for i in range(1, Xp.size):
#     x0 = int(Xp[i - 1])
#     x1 = int(Xp[i])
#     if x1 - x0 > avgL + 2 * stdL:
#       Xzeroes = []
#       currsign = np.sign(W[x0])
#       for i in range(x0 + 1, x1):
#         if currsign != np.sign(W[i]):
#           Xzeroes.append(i)
#           currsign = np.sign(W[i])
#       if len(Xzeroes) > 1:
#         Xnew.append(Xzeroes[0] + e(W[Xzeroes[0] : Xzeroes[-1]]))
#   return np.array(Xnew, dtype=np.int)


# def refine_frontier(Xp, W):
#   "find additional frontier points, and return an array with then, if found"
#   if W[Xp[0]] >= 0:
#     e = np.argmax
#   else:
#     e = np.argmin
#   L = Xp[1:] - Xp[:-1]
#   avgL = np.average(L)
#   stdL = np.std(L)
#   print("std=", stdL)
#   Pulses = []
#   Pulses_to_split = []
#   for i in range(1, Xp.size):
#     x0 = int(Xp[i - 1])
#     x1 = int(Xp[i])
#     if x1 - x0 >= avgL + 2 * stdL:
#       Pulses_to_split.append((x0, x1))
#     # else:
#     #   Pulses_to_split.append((x0, x1))

#   while len(Pulses_to_split) > 0:
#     Pulses_to_test = []
#     for x0, x1 in Pulses_to_split:
#       Xzeroes = []
#       currsign = np.sign(W[x0])
#       for i in range(x0 + 1, x1):
#         if currsign != np.sign(W[i]):
#           Xzeroes.append(i)
#           currsign = np.sign(W[i])
#       if len(Xzeroes) > 1:
#         x = Xzeroes[0] + e(W[Xzeroes[0] : Xzeroes[-1]])
#         Pulses_to_test.append((x0, x))
#         Pulses_to_test.append((x, x1))
#       # else:
#       #   Pulses.append((x0, x1))
#     Pulses_to_split = []
#     for x0, x1 in Pulses_to_test:
#       if x1 - x0 < avgL + 2 * stdL:
#         Pulses.append((x0, x1))
#       else:
#         Pulses_to_split.append((x0, x1))

#   if len(Pulses) > 0:
#     return np.unique(np.array([p[0] for p in Pulses] + [Pulses[-1][1]], dtype=np.int))
#   else:
#     print("No refinement found")
#     return None




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


# def refine_frontier_iter(Xp, W):
#   L = Xp[1:] - Xp[:-1]
#   avgL = np.average(L)
#   stdL = np.std(L)
#   Xp_new = _refine_frontier(Xp, W, avgL, stdL)

#   while not Xp_new is None:
#     Xp_c = np.unique(np.hstack([Xp, Xp_new])).astype(np.int)
#     L = Xp_c[1:] - Xp_c[:-1]
#     stdL_c = np.std(L)
#     if stdL_c <= stdL:
#       avgL = np.average(L)
#       stdL = stdL_c
#       Xp = Xp_c
#       Xp_new = _refine_frontier(Xp, W, avgL, stdL)
#     else:
#       print(f"std0={stdL}, std1={stdL_c}")
#       break
#   return Xp

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