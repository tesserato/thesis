import numpy as np
import random
import wave
from numpy.fft import rfft, irfft
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.linalg import block_diag
from scipy.optimize import nnls
from scipy.signal import savgol_filter
import numpy.polynomial.polynomial as poly
from scipy.signal import hilbert
from scipy.spatial import Delaunay
import plotly.graph_objects as go


def read_wav(path): 
  """returns signal & fps"""
  wav = wave.open(path , 'r')
  signal = np.frombuffer(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps


def save_wav(signal, name = 'test.wav', fps = 44100): 
  '''save .wav file to program folder'''
  o = wave.open(name, 'wb')
  o.setframerate(fps)
  o.setnchannels(1)
  o.setsampwidth(2)
  o.writeframes(np.int16(signal)) # Int16
  o.close()


class Pulse_old:
  def __init__(self, x0, W, is_noise = False):
    self.W = W
    self.start = x0
    self.len = W.size
    self.end = self.start + self.len
    idx = np.argmax(np.abs(W)) #
    self.y = W[idx]            # np.average(W)
    self.x = x0 + idx               # np.sum(np.linspace(0, 1, self.len) * np.abs(W) / np.sum(np.abs(W)))
    self.noise = is_noise


class Pulse:
  def __init__(self, x0, W, is_noise = False):
    # self.W = W
    self.x0 = x0 - .5
    self.len = W.size
    self.x1 = self.x0 + self.len
    idx = np.argmax(np.abs(W)) #
    self.y = W[idx]            # np.average(W)
    self.x = x0 + idx
    self.noise = is_noise


def signal_to_pulses(W):
  ''' returns a list of instances of the Pulses class'''
  n = W.size
  pulses = []
  sign = np.sign(W[0])
  x0 = 0
  for x in range(n):
    if sign != np.sign(W[x]):
      p = Pulse(x0, W[x0 : x])
      pulses.append(p)
      x0 = x
      assert(np.all(np.sign(W[x0 : x]) == sign))
      sign = np.sign(W[x])
  pulses.append(Pulse(x0, W[x0 : n]))
  return pulses

def signal_to_pulses_old(W):
  ''' returns a list of instances of the Pulses class'''
  n = W.size
  pulses = []
  sign = np.sign(W[0])
  x0 = 0
  for x in range(n):
    if sign != np.sign(W[x]):
      p = Pulse(x0, W[x0 : x])
      pulses.append(p)
      x0 = x
      assert(np.all(np.sign(W[x0 : x]) == sign))
      sign = np.sign(W[x])
  pulses.append(Pulse(x0, W[x0 : n]))
  return pulses


def split_pulses(pulses):
  ''' Mark pulses as noise based on Nyquist | divide into positive and negative pulses and noises '''
  pos_pulses = []
  neg_pulses = []
  pos_noises = []
  neg_noises = []
  for p in pulses:
    if p.len <=2:
      p.noise = True
      if p.y >= 0:
        pos_noises.append(p)
      else:
        neg_noises.append(p)
    else:
      if p.y >= 0:
        pos_pulses.append(p)
      else:
        neg_pulses.append(p)
  return pos_pulses, neg_pulses, pos_noises, neg_noises


def get_pulses_area_rectangle(pulses):
  ''' Return X & Y pulses area vectors '''
  X, Y = [], []
  for p in pulses:
    X.append(p.x0)  ; Y.append(0)
    X.append(p.x0 ) ; Y.append(p.y)
    X.append(p.x1)  ; Y.append(p.y)
    X.append(p.x1)  ; Y.append(0)
    X.append(None)  ; Y.append(None)
  return X, Y


def get_pulses_area(pulses):
  ''' Return X & Y pulses area vectors '''
  X, Y = [], []
  for p in pulses:
    X.append(p.x0)  ; Y.append(0)
    X.append(p.x )  ; Y.append(p.y)
    # X.append(p.x1)  ; Y.append(0)
    # X.append(None)  ; Y.append(None)
  X.append(pulses[-1].x1 )  ; Y.append(0)
  return X, Y


def parabola_3_points(X, Y):
  assert len(X) == len(Y) and len(X) == 3
  x0, x1, x2 = X[0], X[1], X[2]
  y0, y1, y2 = Y[0], Y[1], Y[2]

  den = (x2 - x0) * (x0 - x1) * (x1 - x2)

  a = ((x0 - x1) * (y1 - y2) - (x1 - x2) * (y0 - y1)) / (2 * den)
  b = (-x0 * (x0 - x1) * (y1 - y2) + x2 * (x1 - x2) * (y0 - y1)) / den
  c = -a * x0**2 - b * x0 + y0
  return a, b, c

# for i in range(1, len(X) - 1):
#   x0, x1, x2 = (X[i - 1] + X[i]) / 2, X[i], (X[i] + X[i + 1]) / 2
#   y0, y1, y2 = (Y[i - 1] + Y[i]) / 2, Y[i], (Y[i] + Y[i + 1]) / 2

def get_curvature(X, Y, n):
  Y = np.abs(Y)
  pos_curvatures_X = []
  pos_curvatures_Y = []
  neg_curvatures_X = []
  neg_curvatures_Y = []
  for i in range(2, len(X) - 2):
    x0, x1, x2 = (X[i - 2] + X[i - 1]) / 2, X[i], (X[i + 1] + X[i + 2]) / 2
    y0, y1, y2 = (Y[i - 2] + Y[i - 1]) / 2, Y[i], (Y[i + 1] + Y[i + 2]) / 2

    a, b, c = parabola_3_points([x0, x1, x2], [y0, y1, y2])

    t1 = 2*a*x1+b
    t0 = 2*a*x0+b
    avg_curv = (t1 / np.sqrt(t1**2+1) - t0 / np.sqrt(t0**2+1)) / (x1 - x0)

    if avg_curv >= 0:
      pos_curvatures_X.append(x1)
      pos_curvatures_Y.append(avg_curv)
    else:
      neg_curvatures_X.append(x1)
      neg_curvatures_Y.append(avg_curv)

  avg = np.average(pos_curvatures_Y)

  i = 0
  lim = (X[-1] - X[0]) / 4
  while pos_curvatures_X[i] < lim:
    i += 1
  p1 = i
  while pos_curvatures_X[i] < 3 * lim:
    i += 1
  p2 = i

  x0 = 0
  y0 = np.average(pos_curvatures_Y[0 : p1])
  x1 = (X[-1] - X[0]) / 2
  y1 = np.average(pos_curvatures_Y[p1 : p2])
  x2 = n
  y2 = np.average(pos_curvatures_Y[p2 : ])

  max_y = np.max([y0, y1, y2])
  y0, y1, y2 = y0 * avg / max_y, y1 * avg / max_y, y2 * avg / max_y

  #########
  # g = interp1d([x0, x1, x2], [y0, y1, y2], "quadratic")
  # f = lambda x : 1 / g(x)
  #########


  a, b, c = parabola_3_points([x0, x1, x2], [y0, y1, y2])

  # print("abc:",aa, bb, cc)
  # print("xxx:",x0, x1, x2)
  # print("yyy:",y0, y1, y2)

  # def g(x):
  #   x = np.array(x, dtype=np.float64)
  #   return (a * x**2 + b * x + c) #* (avg / cc)

  def radius(x):
    x = np.array(x, dtype=np.float64)
    return 1 / ( (a * x**2 + b * x + c) )#* (avg / cc) )

  return radius


def constrained_least_squares_arbitrary_intervals(X, Y, I, k=3):
  '''constrained least squares with q intervals and k-1 polynomial'''
  assert len(X) == len(Y)
  n = len(X)
  I = [0] + I + [n]

  # U = np.zeros((n, k * q))

  U = []

  for i in range(len(I) - 1):
    l0, l1 = I[i], I[i + 1]
    Ui = np.zeros((l1 - l0, k + 1))
    for j in range(l1 - l0):
      for c in range(k + 1):
        Ui[j, c] = X[l0 + j]**c

    U.append(Ui)

  U = block_diag(*U)

  V = np.zeros((2 * (len(I) - 2), (k + 1) + (k + 1) * (len(I) - 2)))

  for i in range(len(I) - 2):
    V[2 * i, i * (k + 1)] = 1
    V[2 * i, (i + 1) * (k + 1)] = -1
    for c in range(1, k + 1):
      V[2 * i, i * (k + 1) + c] = X[i+1]**c
      V[2 * i + 1, i * (k + 1) + c] = X[i+1]**(c-1)
      V[2 * i, (i + 1) * (k + 1) + c] = -X[i+1]**c
      V[2 * i + 1, (i + 1) * (k + 1) + c] = -X[i+1]**(c-1)

  np.savetxt("U.csv", np.round(U, 2), delimiter=",")
  np.savetxt("V.csv", np.round(V, 2), delimiter=",")

  '''solving'''
  UTUinv = np.linalg.inv(U.T @ U)
  # tau = np.linalg.inv(np.matmul(np.matmul(V, UTUinv),V.T))
  print(V.shape, UTUinv.shape)
  tau = np.linalg.inv(V @ UTUinv @ V.T)
  UTW = np.matmul(U.T, Y)
  par1 = np.matmul(np.matmul(V, UTUinv), UTW)
  A = np.matmul(UTUinv,UTW - np.matmul(np.matmul(V.T, tau),par1))
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


def get_circle(x0, y0, x1, y1, r):
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


def draw_circle(xc, yc, r, fig, n=100):
  '''draws circle as plotly scatter from center and radius'''
  X = []
  Y = []
  for t in np.linspace(0, 2 * np.pi, n):
    X.append(xc + r * np.cos(t))
    Y.append(yc + r * np.sin(t))
  fig.add_trace(
    go.Scatter(
      name="",
      legendgroup="Circles",
      x=X,
      y=Y,
      showlegend=False,
      visible = "legendonly",
      mode="lines",
      line=dict(
          width=1,
          color="green"
      )
    )
  )
  return X, Y


def get_frontier_old(X, Y, nn):
  '''extracts the envelope via snowball method'''
  radius = get_curvature(X, Y, nn)
  n = len(X)
  idx1 = 0
  idx2 = 1
  frontier_X = [0]
  frontier_Y = [Y[0]]
  while idx2 < n:
    r = radius((X[idx1] + X[idx2]) / 2)
    xc, yc = get_circle(X[idx1], Y[idx1], X[idx2], Y[idx2], r)
    empty = True
    for i in range(idx2 + 1, n):
      if np.sqrt((xc - X[i])**2 + (yc - Y[i])**2) < r:
        empty = False
        idx2 += 1
        break
    if empty:
      frontier_X.append(X[idx2])
      frontier_Y.append(Y[idx2])
      idx1 = idx2
      idx2 += 1
      # draw_circle(xc, yc, r, fig)
  frontier_X.append(nn)
  frontier_Y.append(Y[-1])
  f = interp1d(frontier_X, frontier_Y, kind="linear")
  X = np.arange(nn)
  return X, f(X), radius(X)


def get_curvature_function(X, Y):
  X = np.array(X)
  Y = np.array(Y)
  # avg_Y = np.average(Y)
  # Y = Y - avg_Y
  m0 = np.average(Y[1:] - Y[:-1]) / np.average(X[1:] - X[:-1])
  curvatures_X = []
  curvatures_Y = []
  for i in range(len(X) - 1):
    m1 = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])

    theta = np.arctan( (m1 - m0) / (1 + m1 * m0) )
    k = np.sin(theta) / (X[i + 1] - X[i])

    # if k >= 0:
    curvatures_X.append((X[i + 1] + X[i]) / 2)
    curvatures_Y.append(k)
    
  curvatures_Y = np.array(curvatures_Y)
  coefs = poly.polyfit(curvatures_X, curvatures_Y, 0)
  smooth_curvatures_Y = poly.polyval(curvatures_X, coefs)

  k_of_x = interp1d(curvatures_X, 2/np.abs(smooth_curvatures_Y), fill_value="extrapolate")

  return k_of_x, curvatures_X, curvatures_Y, curvatures_X, smooth_curvatures_Y


def get_frontier(Pulses):
  '''extracts the envelope via snowball method'''

  # fig.add_trace(
  #   go.Scatter(
  #     name="Circles",
  #     legendgroup="Circles",
  #     x=[0],
  #     y=[0],
  #     mode="none",
  #     visible = "legendonly",
  #     line=dict(
  #         width=1,
  #         color="green"
  #     )
  #   )
  # )

  r_of_x, _, _, _, _ = get_curvature_function([p.x for p in Pulses], [p.y for p in Pulses])
  idx1 = 0
  idx2 = 1
  Frontier = [Pulses[0]]
  n = len(Pulses)
  while idx2 < n:
    r = r_of_x((Pulses[idx1].x + Pulses[idx2].x) / 2)
    # xc, yc = get_circle(Pulses[idx1].x1, Pulses[idx1].y, Pulses[idx2].x0, Pulses[idx2].y, r) # Square
    xc, yc = get_circle(Pulses[idx1].x, Pulses[idx1].y, Pulses[idx2].x, Pulses[idx2].y, r) # Triangle
    empty = True
    for i in range(idx2 + 1, n):
      if np.sqrt((xc - Pulses[i].x0)**2 + (yc - Pulses[i].y)**2) < r:
        empty = False
        idx2 += 1
        break
    if empty:

      # draw_circle(xc, yc, r, fig)

      Frontier.append(Pulses[idx2])
      idx1 = idx2
      idx2 += 1
  Frontier.append(Pulses[-1])
  # f = interp1d(frontier_X, frontier_Y, kind="linear")
  # X = np.arange(nn)
  return [p.x for p in Frontier], [p.y for p in Frontier]


#####################################################################################################################
################################################### OLD FUNCTIONS ###################################################
#####################################################################################################################

def get_delaunay(X, Y):
  points = np.array([[x, y] for x, y in zip(pos_X, pos_Y)])
  tri = Delaunay(points)
  X_tri = []
  Y_tri = []
  for ia, ib, ic in tri.vertices:
    xa, ya = points[ia]
    xb, yb = points[ib]
    xc, yc = points[ic]
    X_tri.append(xa), Y_tri.append(ya)
    X_tri.append(xb), Y_tri.append(yb)
    X_tri.append(xc), Y_tri.append(yc)
    X_tri.append(xa), Y_tri.append(ya)
    X_tri.append(None), Y_tri.append(None)
  return X_tri, Y_tri

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n, 2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def get_curvature_3_points(X, Y):
  '''Return max curvatures, avg and r for the poly approximation of 3 pulses amplitude at a time'''
  curvatures = []
  for i in range(len(X) - 2):
    x0, x1, x2 = X[i], X[i + 1], X[i + 2]
    y0, y1, y2 = Y[i], Y[i + 1], Y[i + 2]
    x, y = np.array([x0, x1, x2]), np.array([y0, y1, y2])
    # y = y / y1
    a0, a1, a2 = poly.polyfit(x, y, 2)
    # c = np.abs(2 * a2) # max curvature
    c = ((2 * a2 * x1 + a1) / np.sqrt((2 * a2 * x1 + a1)**2 + 1) - (2 * a2 * x0 + a1) / np.sqrt((2 * a2 * x0 + a1)**2 + 1)) / (x1-x0)
    # c = 2 * a2 / ((a1 + 2*a2*x1)**2 + 1)**(3/2)
    curvatures.append(c)
    # X = np.linspace(x0, x2, 100)
    # Y = poly.polyval(X, args)
    # for x, y in zip(X, Y):
    #   pos_curvature_X.append(x); pos_curvature_Y.append(y)
    # pos_curvature_X.append(None) ; pos_curvature_Y.append(None)
  average_curvature = np.abs(np.average(curvatures))
  radius = 1 / average_curvature
  curvatures = [curvatures[0]] + curvatures + [curvatures[-1]]
  return curvatures, average_curvature, radius

def smoothstep(t):

  # return 3 * t**2 - 2 * t**3
  # return 6 * t**5 - 15 * t**4 + 10 * t**3
  # return -20 * t**7 + 70 * t**6 - 84 * t**5 + 35 * t**4
  return t

def old_smooth(X, Y, n):

  Yn = np.zeros(len(Y))

  Yn[0] = (2 * np.sqrt(Y[0]) + np.sqrt(Y[1]))**2 / 9
  Yn[-1] = (2 * np.sqrt(Y[-1]) + np.sqrt(Y[-2]))**2 / 9
  
  for i in range(1, len(Y) - 1):
    Yn[i] = (np.sqrt(Y[i - 1]) + np.sqrt(Y[i]) + np.sqrt(Y[i +1]))**2 / 9
  Y[:] = Yn[:]
  
  X[0] = 0
  Xs = np.arange(n)
  Ye0 = np.zeros(n)
  Ye1 = np.zeros(n)
  Yo0 = np.zeros(n)
  Yo1 = np.zeros(n)

  t = np.linspace(0, .5, X[1])[::-1]
  w = smoothstep(t)
  Yo1[0 : X[1]] = w * Y[0]
  Yo0[0 : X[1]] = (1 - w) * Y[1]

  for i in range(0, len(X) - 5, 4):
    x0, x1, x2, x3, x4, x5 = X[i], X[i + 1], X[i + 2], X[i + 3], X[i + 4], X[i + 5]
    y0, y1, y2, y3, y4, y5 = Y[i], Y[i + 1], Y[i + 2], Y[i + 3], Y[i + 4], Y[i + 5]

    '''even'''
    t = np.linspace(0, 1, x2 - x0)
    w = smoothstep(t)
    Ye0[x0 : x2] = y0 * (1 - w)
    Ye1[x0 : x2] = y2 * w
    
    t = np.linspace(0, 1, x4 - x2)[::-1]
    w = smoothstep(t)
    Ye0[x2 : x4] = y4 * (1 - w)
    Ye1[x2 : x4] = y2 * w
    
    '''odd'''
    t = np.linspace(0, 1, x3 - x1)
    w = smoothstep(t)
    Yo0[x1 : x3] = y1 * (1 - w)
    Yo1[x1 : x3] = y3 * w
    
    t = np.linspace(0, 1, x5 - x3)[::-1]
    w = smoothstep(t)
    Yo0[x3 : x5] = y5 * (1 - w)
    Yo1[x3 : x5] = y3 * w
    

  t = np.linspace(0, .5, n - X[-4])
  w = smoothstep(t)
  Yo1[X[-4] : ] = w * Y[-1]
  Yo0[X[-4] : ] = (1 - w) * Y[-4]

  t = np.linspace(0, 1, n - X[-5])
  w = smoothstep(t)
  Ye1[X[-5] : ] = w * Y[-2]
  Ye0[X[-5] : ] = (1 - w) * Y[-5]

  YY = (Ye0 + Ye1 + Yo0 + Yo1) / 2
  return Xs, YY, Ye0, Ye1, Yo0, Yo1

def fit_nonnegative_line(X, Y):
  '''y = ax + b'''
  Y = np.abs(Y)

  prob = pulp.LpProblem("LP", pulp.LpMinimize)

  a = pulp.LpVariable("a")
  b = pulp.LpVariable("b", 0)
  
  Z = []
  for i in range(len(X)):
    z_i = pulp.LpVariable(f"z_{i}", 0)
    prob +=   Y[i] - (a * X[i] + b) <= z_i
    prob += -(Y[i] - (a * X[i] + b)) <= z_i
    Z.append(z_i)
  prob += a * X[-1] + b >= 0
  prob += pulp.lpSum([z for z in Z]) 

  prob.writeLP("lp.lp")

  status = prob.solve(pulp.CPLEX_CMD(path=r"C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\bin\x64_win64\cplex.exe"))

  print(pulp.LpStatus[status])
  a = pulp.value(a)
  b = pulp.value(b)
  print(a, b)
  return a, b

def get_average_curvature(X, Y):
  '''Return average curvature and radius of the equivalent circle for the poly approximation of 4 points amplitude at a time'''
  Ycurvatures = []
  Xcurvatures = []
  for i in range(len(X) - 3):
    x0, x1, x2, x3 = X[i], X[i + 1], X[i + 2], X[i + 3]
    y0, y1, y2, y3 = Y[i], Y[i + 1], Y[i + 2], Y[i + 3]
    x, y = np.array([x0, x1, x2, x3]), np.array([y0, y1, y2, y3])
    a0, a1, a2 = poly.polyfit(x, y, 2)
    xc = (x0 + x1 + x2 + x3) / 4
    yc = 2 * a2 / ((a1 + a2 * xc)**2 + 1)**(3/2) # curvature at x = (x1+x2)/2

    # t1 = 2*a2*x3+a1 # average curvature, as per integral, at x = (x1+x2)/2
    # t0 = 2*a2*x0+a1
    # c = (t1 / np.sqrt(t1**2+1) - t0 / np.sqrt(t0**2+1)) / (x3 - x0)

    Ycurvatures.append(yc)
    Xcurvatures.append(xc)

  # k = 3
  # A = np.zeros((len(Ycurvatures), 3))
  # for l in range(len(Ycurvatures)):
  #   A[l, 0] = 1
  #   A[l, 1] = Xcurvatures[l]
  #   A[l, 2] = 0
  # A[-1, -1] = -1
  # coefs, _ = nnls(A, np.abs(Ycurvatures))
  # coefs = np.linalg.lstsq(A, np.abs(Ycurvatures))[0]
  # print(coefs)
  # b, a, slack = coefs
  # curvatures = poly.polyval(Xcurvatures, [b, a])

  a, b = fit_nonnegative_line(Xcurvatures, Ycurvatures)

  radius = lambda x : 1 / (a * x + b)

  fig.add_trace(
    go.Scatter(
      name="curvatures",
      x=Xcurvatures,
      y=np.abs(Ycurvatures),
      mode="markers",
      marker=dict(
          # size=8,
          color="yellow",
          # showscale=False
      )
    )
  )

  fig.add_trace(
    go.Scatter(
      name="curvatures appr",
      x=Xcurvatures,
      y=a * np.array(Xcurvatures) + b,
      mode="lines",
      line=dict(
          # size=8,
          color="pink",
          # showscale=False
      )
    )
  )

  fig.add_trace(
    go.Scatter(
      name="curvatures appr",
      x=[0],
      y=[np.average(np.abs(Ycurvatures))],
      mode="markers",
      marker=dict(
          # size=8,
          color="green",
          # showscale=False
      )
    )
  )

  return radius

def get_average_curvature_direct(X, Y):
  '''Return average curvature and radius of the equivalent circle for the poly approximation of 4 points amplitude at a time'''
  curvatures = []
  for i in range(2, len(X) - 2):
    x0, x1, x2, x3, x4 = X[i - 2], X[i - 1], X[i], X[i + 1], X[i + 2]
    y0, y1, y2, y3, y4 = Y[i - 1], Y[i - 1], Y[i], Y[i + 1], Y[i + 1]

    yl1 = (y2 - y0) / (x2 - x0)
    yl2 = (y3 - y1) / (x3 - x1)
    yl3 = (y4 - y2) / (x4 - x2)

    yll1 = (yl2 - yl1) / (x2 - x1)
    yll2 = (yl3 - yl2) / (x3 - x2)

    yl = (yl1 + yl2 + yl3) / 3
    yll = (yll1 + yll2) / 2

    c = yll / (1 + yl**2)**(3 / 2)

    curvatures.append(c)

  a, b = fit_nonnegative_line(X[2 : -2], np.abs(curvatures))

  radius = lambda x : 1 / np.abs(np.average(curvatures)) # (a * x + b)

  fig.add_trace(
    go.Scatter(
      name="curvatures",
      x=X[1 : -1],
      y=curvatures,
      mode="markers",
      marker=dict(
          # size=8,
          color="yellow",
          # showscale=False
      )
    )
  )

  fig.add_trace(
    go.Scatter(
      name="curvatures appr",
      x=X[1 : -1],
      y=a * np.array(curvatures) + b,
      mode="lines",
      line=dict(
          # size=8,
          color="pink",
          # showscale=False
      )
    )
  )

  fig.add_trace(
    go.Scatter(
      name="curvatures appr",
      x=[0],
      y=[np.average((curvatures))],
      mode="markers",
      marker=dict(
          # size=8,
          color="green",
          # showscale=False
      )
    )
  )

  return radius

def constrained_least_squares(X, Y, q=4, k=4):
  '''constrained least squares with q intervals and k-1 polynomial'''
  assert X.size == Y.size
  n = X.size
  l = n // q
  U = np.zeros((n, k * q))
  V = np.zeros((2 * (q-1), k * q))

  '''populating U'''
  for i1 in range(q):
    for i2 in range(l * i1, l * (i1 + 1)):
      for i3 in range(k):
        U[i2, i1 * k + i3] = X[i2]**i3

  '''populating V (constraints)'''
  for i in range(q - 1):
    for j in range(k):
      V[2 * i, i * k + j] = X[(i + 1) * l]**j
      V[2 * i, i * k + k + j] = -X[(i + 1) * l]**j
  for i in range(q - 1):
    for j in range(1, k):
      V[2 * i + 1, i * k + j] = j * X[(i + 1) * l] ** (j - 1)
      V[2 * i + 1, i * k + k + j] = -j * X[(i + 1) * l] ** (j - 1)

  '''solving'''
  UTUinv = np.linalg.inv(np.matmul(U.T, U))
  tau = np.linalg.inv(np.matmul(np.matmul(V, UTUinv),V.T))
  UTW = np.matmul(U.T, Y)
  par1 = np.matmul(np.matmul(V, UTUinv), UTW)
  A = np.matmul(UTUinv,UTW - np.matmul(np.matmul(V.T, tau),par1))
  return np.reshape(A, (q, -1))

def coefs_to_array(A, X): 
  '''evaluates x E X from a coefficient matrix A'''
  n = X.size
  q = A.shape[0]
  k = A.shape[1]
  l = n // q
  Y = []
  for i in range(q):
    for x in range(i * l, (i + 1) * l):
      y = 0
      for j in range(k):
        y += A[i, j] * X[x]**j
      Y.append(y)

  x = len(Y)
  while x < n:
    y = 0
    for i in range(k):
      y += A[q-1, i] * X[x]**i
    Y.append(y)
    x += 1
  return Y

def remove_envelope(X, Y, pulses):
  W = np.zeros(pulses[-1].start + pulses[-1].len)
  f = interp1d((X), Y, kind="linear", fill_value='extrapolate', assume_sorted=True)
  # f = UnivariateSpline(X, Y)
  for p in pulses:
    x0 = p.start
    x1 = p.start + p.len
    Xn = p.start + np.arange(p.len)
    A = f(Xn)
    W[x0 : x1] = p.W / A
  return W, f

def smooth(X, Y, n):
  assert len(X) == len(Y), f"len(X)={len(X)} != len(Y)={len(Y)}"
  Xs = np.zeros(len(X))
  Ys = np.zeros(len(X))
  Xs[0], Xs[-1] = 0, n
  Ys[0], Ys[-1] = Y[0], Y[-1]
  
  for i in range(1, len(X) - 1):
    Xs[i] = (X[i - 1] + X[i] + X[i + 1]) / 3
    Ys[i] = (Y[i - 1] + Y[i] + Y[i + 1]) / 3
  # print(Xs)
  # f = interp1d(Xs, Ys, kind="quadratic")
  f = UnivariateSpline(Xs, Ys)
  Xf = np.arange(n)
  Yf = f(Xf)

  return Xf, Yf

def line(x0, x1, y0, y1, t):
  '''y = m x + b | 0 = x0, 1 = x1'''
  x = x0 + t * (x1 - x0)
  m = (y1 - y0) / (x1 - x0)
  b = y0 - m * x0
  f = lambda alpha : m * alpha + b
  y = f(x)
  return x, y

def bezier_approximation(Xo, Yo, n):
  Xp = []
  Yp = []
  Xp.append(Xo[0])
  Yp.append(Yo[0])
  for i in range(1, len(Xo)-2):
    Xp.append(Xo[i]), Yp.append(Yo[i])
    Xp.append((Xo[i] + Xo[i + 1]) / 2), Yp.append((Yo[i] + Yo[i + 1]) / 2)
  Xp.append(Xo[-2])
  Yp.append(Yo[-2])
  Xp.append(Xo[-1])
  Yp.append(Yo[-1])
  # print(Xp)

  X = []
  Y = []
  for i in range(0, len(Xp) - 3, 2):
    distance = np.sqrt((Xp[i + 2] - Xp[i])**2 + (Yp[i + 2] - Yp[i])**2)
    distance = np.int(np.ceil(distance))
    mt = 1 * (distance - 1) / distance
    for t in np.linspace(0, mt, distance):
      x0, y0 = line(Xp[i], Xp[i + 1], Yp[i], Yp[i + 1], t)
      x1, y1 = line(Xp[i + 1], Xp[i + 2], Yp[i + 1], Yp[i + 2], t)
      x, y = line(x0, x1, y0, y1, t)
      X.append(x)
      Y.append(y)
  distance = np.sqrt((Xp[-1] - Xp[-3])**2 + (Yp[-1] - Yp[-3])**2)
  distance = np.int(np.ceil(distance))
  for t in np.linspace(0, 1, distance):
    x0, y0 = line(Xp[-3], Xp[-2], Yp[-3], Yp[-2], t)
    x1, y1 = line(Xp[-2], Xp[-1], Yp[-2], Yp[-1], t)
    x, y = line(x0, x1, y0, y1, t)
    X.append(x)
    Y.append(y)

  f = interp1d(X, Y, kind="cubic", fill_value="extrapolate", assume_sorted=True)
  X = np.arange(n)
  Y = f(X)
  return X, Y

def get_knots_indices(X, Y, knots=10):
  Y = np.abs(Y)
  A = np.sum(Y) / (knots + 1)
  Xa = []
  Ya = []
  i = 0
  I = []
  while len(I) < knots:
    S = 0
    Xinterm = []
    Yinterm = []
    while S < A:
      S += Y[i]
      Xinterm.append(X[i])
      Yinterm.append(Y[i])
      i += 1
    I.append(i)
    Xa.append(np.average(Xinterm))
    Ya.append(np.average(Yinterm))
  Xa.append(np.average(X[i:]))
  Ya.append(np.average(Y[i:]))

  f = interp1d(Xa, Ya, "cubic", fill_value="extrapolate")

  print(I)
  return X, f(X), I

def get_curvature_direct(X, Y, n):
  Y = np.abs(Y)
  '''Return average curvature and radius of the equivalent circle for the poly approximation of 4 points amplitude at a time'''
  pos_curvatures_X = []
  pos_curvatures_Y = []
  neg_curvatures_X = []
  neg_curvatures_Y = []
  for i in range(2, len(X) - 2):
    x0, x1, x2, x3, x4 = X[i - 2], X[i - 1], X[i], X[i + 1], X[i + 2]
    y0, y1, y2, y3, y4 = Y[i - 2], Y[i - 1], Y[i], Y[i + 1], Y[i + 2]

    yl1 = (y2 - y0) / (x2 - x0)
    yl2 = (y3 - y1) / (x3 - x1)
    yl3 = (y4 - y2) / (x4 - x2)

    yll1 = (yl2 - yl1) / (x2 - x1)
    yll2 = (yl3 - yl2) / (x3 - x2)

    yl = (yl1 + yl2 + yl3) / 3
    yll = (yll1 + yll2) / 2

    c = yll / (1 + yl**2)**(3 / 2)
    if c >= 0:
      pos_curvatures_X.append(x2)
      pos_curvatures_Y.append(c)
    else:
      neg_curvatures_X.append(x2)
      neg_curvatures_Y.append(c)

  i = 0
  lim = (X[-1] - X[0]) / 4
  while pos_curvatures_X[i] < lim:
    i += 1
  p1 = i
  while pos_curvatures_X[i] < 3 * lim:
    i += 1
  p2 = i

  x0 = 0
  y0 = np.average(pos_curvatures_Y[0 : p1])
  x1 = (X[-1] - X[0]) / 2
  y1 = np.average(pos_curvatures_Y[p1 : p2])
  x2 = n
  y2 = np.average(pos_curvatures_Y[p2 : ])

  #########
  # g = interp1d([x0, x1, x2], [y0, y1, y2], "quadratic")
  # f = lambda x : 1 / g(x)
  #########

  avg = np.average(pos_curvatures_Y)

  aa, bb, cc = parabola_3_points([x0, x1, x2], [y0, y1, y2])

  # print("abc:",aa, bb, cc)
  # print("xxx:",x0, x1, x2)
  # print("yyy:",y0, y1, y2)

  def g(x):
    x = np.array(x, dtype=np.float64)
    return (aa * x**2 + bb * x + cc) * (avg / cc)

  def f(x):
    x = np.array(x, dtype=np.float64)
    return 1 / ( (aa * x**2 + bb * x + cc) * (avg / cc) )

  return f

def bezier_3_points(X, Y):
  assert len(X) == len(Y) and len(X) == 3

  # def line(x0, x1, y0, y1, t):
  #   '''y = m x + b | 0 = x0, 1 = x1'''
  #   x = x0 + t * (x1 - x0)
  #   m = (y1 - y0) / (x1 - x0)
  #   b = y0 - m * x0
  #   f = lambda alpha : m * alpha + b
  #   y = f(x)
  #   return x, y

  x0, x1, x2 = X[0], X[1], X[2]
  # y0, y1, y2 = Y[0], Y[1], Y[2]

  if x0 + x2 == 2 * x1:
    print("oooops")
    x2 = x2 + 0.001

  def bez(x, x0=X[0], x1=X[1], x2=x2, y0=Y[0], y1=Y[1], y2=Y[2]):
    a = x0 - x1
    b = np.sqrt(x * x0 - 2 * x * x1 + x * x2 - x0 * x2 + x1**2)
    c = x0 - 2 * x1 + x2
    # print(">>>", a, b, c)
    t = (a+b) / c
    # xa, ya = line(x0, x1, y0, y1, t)
    # xb, yb = line(x1, x2, y1, y2, t)
    # x, y = line(xa, xb, ya, yb, t)
    y = y0 * (1 - t)**2 + 2 * t * (1 - t) * y1 + y2 * t**2
    return y

  return bez

def cubic_3_points(X, Y):
  '''y = a * x**3 + b * x**2 + c * x + d'''
  assert len(X) == len(Y) and len(X) == 3
  x0, x1, x2 = X[0], X[1], X[2]
  y0, y1, y2 = Y[0], Y[1], Y[2]

  den = (x0 - x1) * (x0 - x2)**3 * (x1 - x2)
  ter = (x0 * y1 - x0 * y2 - x1 * y0 + x1 * y2 + x2 * y0 - x2 * y1)
  a = (x0 - 2 * x1 + x2) * ter / den
  b = -(2 * x0**2 - 3 * x0*x1 + 2 * x0 * x2 - 3 * x1 * x2 + 2 * x2**2) * ter / den
  c = (-a*x0**3 + a*x2**3 - b*x0**2 + b*x2**2 + y0 - y2)/(x0 - x2)
  d = y0 - (a * x0**3 + b * x0**2 + c * x0)
  return a, b, c, d

def get_discrete_curvature_old_old(X, Y):
  radius_X = []
  radius_Y = []
  for i in range(1, len(X) - 1):
    x0, x1, x2 = (X[i - 1] + X[i]) / 2, X[i], (X[i] + X[i + 1]) / 2
    y0, y1, y2 = (Y[i - 1] + Y[i]) / 2, Y[i], (Y[i] + Y[i + 1]) / 2

    # k = np.arccos(((x0 - x1)*(x1 - x2) + (y0 - y1)*(y1 - y2))/(np.sqrt((x0 - x1)**2 + (y0 - y1)**2)*np.sqrt((x1 - x2)**2 + (y1 - y2)**2)))/(np.sqrt((x0/2 - x1/2)**2 + (y0/2 - y1/2)**2) + np.sqrt((x1/2 - x2/2)**2 + (y1/2 - y2/2)**2))

    # k = np.arccos(((-x0/2 + x1/2)*(-x1/2 + x2/2) + (-y0/2 + y1/2)*(-y1/2 + y2/2))/(np.sqrt((x0/2 - x1/2)**2 + (y0/2 - y1/2)**2)*np.sqrt((x1/2 - x2/2)**2 + (y1/2 - y2/2)**2)))/(np.sqrt((x0/2 - x1/2)**2 + (y0/2 - y1/2)**2) + np.sqrt((x1/2 - x2/2)**2 + (y1/2 - y2/2)**2))

    k = -np.arctan(((x0 - x1)*(y1 - y2) + (x1 - x2)*(-y0 + y1))/((x0 - x1)*(x1 - x2) + (y0 - y1)*(y1 - y2)))

    # if k >= 0:
    radius_X.append(x1)
    radius_Y.append(k)

  # radius_Y = np.array(radius_Y)
  # radius_Y = radius_Y / np.average(radius_Y)
      
  # coefs = poly.polyfit(radius_X, radius_Y, 2)
  # radius_Y = poly.polyval(radius_X, coefs)

  return radius_X, radius_Y

def get_discrete_curvature_old(X, Y, segments=5):
  lim = np.sum(Y) / segments
  X_idxs = []
  cr_idx = 0
  for i in range(segments - 1):
    cr_sum = 0
    while cr_sum < lim:
      cr_sum += Y[cr_idx]
      cr_idx += 1
    X_idxs.append(cr_idx)
  XX = []
  YY = []
  XCC = []
  YCC = []
  X_idxs = [0] + X_idxs + [len(X) - 1]
  for i in range(len(X_idxs) - 1):
    idx0 = X_idxs[i]
    idx1 = X_idxs[i + 1]
    segmentX = np.arange(int(X[idx0]), int(X[idx1]), 1, dtype=np.float64)
    c, b, a = poly.polyfit(X[idx0 : idx1], Y[idx0 : idx1], 2)
    if a > 0:
      XCC.append((X[idx0] + X[idx1]) / 2)
      YCC.append(2 * a)

    segmentY = a * segmentX**2 + b * segmentX + c
    for x, y in zip(segmentX, segmentY):
      XX.append(x)
      YY.append(y)

  f = interp1d(XCC, 1 / np.abs(YCC), fill_value="extrapolate")
      
  return XX, f(XX), XCC, 1 / np.abs(YCC), f

def get_curvature_function_(X, Y):
  X = np.array(X, dtype=np.float64)
  Y = np.array(Y, dtype=np.float64)

  curvatures_X = X[:-1] + (X[1:] - X[:-1]) / 2
  curvatures_Y = np.sqrt((Y[1: ] - Y[ :-1])**2 + (X[1: ] - X[ :-1])**2) * 100

  # curvatures_Y = np.array(curvatures_Y)
  coefs = poly.polyfit(curvatures_X, curvatures_Y, 0)
  smooth_curvatures_Y = poly.polyval(curvatures_X, coefs)

  k_of_x = interp1d(curvatures_X, smooth_curvatures_Y, fill_value="extrapolate")

  return k_of_x, curvatures_X, curvatures_Y, curvatures_X, smooth_curvatures_Y

def get_pulses_area_triangle(pulses):
  ''' Return X & Y pulses area vectors '''
  X, Y = [], []
  for p in pulses:
    X.append(p.start - .5); Y.append(0)
    X.append(p.x )        ; Y.append(p.y)
    # X.append(p.end - .5)  ; Y.append(0)
    # X.append(p.start - .5); Y.append(0)
    # X.append(None)        ; Y.append(None)
  X.append(pulses[-1].end - .5)  ; Y.append(0)
  return X, Y