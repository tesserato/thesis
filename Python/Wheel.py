import numpy as np
from scipy.interpolate import interp1d
import numpy.polynomial.polynomial as poly


def _get_circle(x0, y0, x1, y1, r):
  '''returns center of circle that passes through two points'''
  
  radsq = r * r
  q = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

  # assert q <= 2 * r, f"shit"

  x3 = (x0 + x1) / 2
  y3 = (y0 + y1) / 2

  if y0 + y1 >= 0:
    xc = x3 + np.sqrt(radsq - (q / 2)**2) * ((y0 - y1) / q)
    yc = y3 + np.sqrt(radsq - (q / 2)**2) * ((x1 - x0) / q)
  else:
    xc = x3 - np.sqrt(radsq - (q / 2)**2) * ((y0 - y1) / q)
    yc = y3 - np.sqrt(radsq - (q / 2)**2) * ((x1 - x0) / q)

  return xc, yc


def _get_radius_function(X, Y):
  m0 = np.average(Y[1:] - Y[:-1]) / np.average(X[1:] - X[:-1])
  curvatures_X = []
  curvatures_Y = []
  # mm = np.sqrt(m0**2 + 1)                            # 1
  for i in range(len(X) - 1):
    # x = (X[i + 1] - X[i])                            # 1
    # y = (Y[i + 1] - Y[i])                            # 1
    # k = -(m0 * x - y) / (x * mm * np.sqrt(x*x + y*y))# 1

    m1 = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])     # 2
    theta = np.arctan( (m1 - m0) / (1 + m1 * m0) ) # 2
    k = np.sin(theta) / (X[i + 1] - X[i])          # 2

    curvatures_X.append((X[i + 1] + X[i]) / 2)
    curvatures_Y.append(k)
    
  curvatures_Y = np.array(curvatures_Y)

  coefs = poly.polyfit(curvatures_X, curvatures_Y, 0)                                       # a
  smooth_curvatures_Y = poly.polyval(curvatures_X, coefs)                                   # a
  r_of_x = interp1d(curvatures_X, 1 / np.abs(smooth_curvatures_Y), fill_value="extrapolate")# a

  return r_of_x


def _get_radius_average(X, Y):
  m0 = np.average(Y[1:] - Y[:-1]) / np.average(X[1:] - X[:-1])
  # curvatures_X = []
  # curvatures_Y = []
  k_sum = 0
  # mm = np.sqrt(m0**2 + 1)                            # 1
  for i in range(len(X) - 1):
    # x = (X[i + 1] - X[i])                            # 1
    # y = (Y[i + 1] - Y[i])                            # 1
    # k = -(m0 * x - y) / (x * mm * np.sqrt(x*x + y*y))# 1

    m1 = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])     # 2
    theta = np.arctan( (m1 - m0) / (1 + m1 * m0) ) # 2
    k = np.sin(theta) / (X[i + 1] - X[i])          # 2

    # curvatures_X.append((X[i + 1] + X[i]) / 2)
    # curvatures_Y.append(k)

    k_sum += k
    
  # curvatures_Y = np.array(curvatures_Y)

  # coefs = poly.polyfit(curvatures_X, curvatures_Y, 0)                                       # a
  # smooth_curvatures_Y = poly.polyval(curvatures_X, coefs)                                   # a
  # r_of_x = interp1d(curvatures_X, 1 / np.abs(smooth_curvatures_Y), fill_value="extrapolate")# a

  r = 1 / (k_sum / (len(X) - 1))    #b
  def r_of_x(x):                  #b
    return r                      #b

  return r_of_x


def _get_pulses(W):
  '''Sorts a signal into pulses, returning positive and negative X, Y coordinates, and filtering out noise'''
  n = W.size
  sign = np.sign(W[0])
  x = 1
  while np.sign(W[x]) == sign:
    x += 1
  x0 = x + 1
  sign = np.sign(W[x0])

  posX = []
  posY = []
  negX = []
  negY = []
  for x in range(x0, n):
    if np.sign(W[x]) != sign: # Prospective pulse
      if x - x0 > 2:          # Not noise
        xp = x0 + np.argmax(np.abs(W[x0 : x]))
        yp = W[xp]
        if np.sign(yp) >= 0:
          posX.append(xp)
          posY.append(yp)
        else:
          negX.append(xp)
          negY.append(yp)
      x0 = x
      sign = np.sign(W[x])

  return np.array(posX), np.array(posY), np.array(negX), np.array(negY)


def _get_frontier(X, Y):
  '''extracts the frontier via snowball method'''
  scaling = (np.average(X[1:]-X[:-1]) / 2) / np.average(Y)
  Y = Y * scaling

  r_of_x = _get_radius_average(X, Y)
  idx1 = 0
  idx2 = 1
  frontierX = [X[0]]
  frontierY = [Y[0]]
  n = len(X)
  while idx2 < n:
    r = r_of_x((X[idx1] + X[idx2]) / 2)
    xc, yc = _get_circle(X[idx1], Y[idx1], X[idx2], Y[idx2], r)
    empty = True
    for i in range(idx2 + 1, n):
      if np.sqrt((xc - X[i])**2 + (yc - Y[i])**2) < r:
        empty = False
        idx2 += 1
        break
    if empty:
      frontierX.append(X[idx2])
      frontierY.append(Y[idx2])
      idx1 = idx2
      idx2 += 1
  frontierX = np.array(frontierX)
  frontierY = np.array(frontierY) / scaling
  return frontierX, frontierY


def frontiers(W):
  "Returns positive and negative frontiers of a signal"
  PosX, PosY, NegX, NegY = _get_pulses(W)
  PosFrontierX, PosFrontierY = _get_frontier(PosX, PosY)
  NegFrontierX, NegFrontierY = _get_frontier(NegX, NegY)
  return PosFrontierX, PosFrontierY, NegFrontierX, NegFrontierY





