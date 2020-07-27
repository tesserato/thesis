import plotly.graph_objects as go
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.linalg import block_diag
from scipy.optimize import nnls
import numpy as np
from Helper import read_wav
from scipy.signal import savgol_filter
import numpy.polynomial.polynomial as poly
import pulp


class Pulse:
  def __init__(self, x0, W, is_noise = False):
    self.W = W
    self.start = x0
    self.len = W.size
    self.end = self.start + self.len
    idx = np.argmax(np.abs(W)) #
    self.y = W[idx]            # np.average(W)
    self.x = x0 + idx               # np.sum(np.linspace(0, 1, self.len) * np.abs(W) / np.sum(np.abs(W)))
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


def get_pulses_area(pulses):
  ''' Return X & Y pulses area vectors '''
  X, Y = [], []
  for p in pulses:
    X.append(p.start - .5); Y.append(0)
    X.append(p.start - .5); Y.append(p.y)
    X.append(p.end - .5)  ; Y.append(p.y)
    X.append(p.end - .5)  ; Y.append(0)
    X.append(None)        ; Y.append(None)
  return X, Y


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


def get_curvature(X, Y, n):
  '''Return average curvature and radius of the equivalent circle for the poly approximation of 4 points amplitude at a time'''
  pos_curvatures_X = []
  pos_curvatures_Y = []
  neg_curvatures_X = []
  neg_curvatures_Y = []
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
    if c >= 0:
      pos_curvatures_X.append(x2)
      pos_curvatures_Y.append(c)
    else:
      neg_curvatures_X.append(x2)
      neg_curvatures_Y.append(c)

  if np.average(pos_curvatures_Y) < np.abs(np.average(neg_curvatures_Y)):
    curvatures_X = pos_curvatures_X
    curvatures_Y = pos_curvatures_Y
  else:
    curvatures_X = neg_curvatures_X
    curvatures_Y = np.abs(neg_curvatures_Y)

  i = 0
  lim = (X[-1] - X[0]) / 3
  while curvatures_X[i] < lim:
    i += 1
  p1 = i
  while curvatures_X[i] < 2 * lim:
    i += 1
  p2 = i

  x0 = 0
  y0 = np.average(curvatures_Y[0 : p1])
  x1 = (X[-1] - X[0]) / 2
  y1 = np.average(curvatures_Y[p1 : p2])
  x2 = n
  y2 = np.average(curvatures_Y[p2 : ])

  #########
  # g = interp1d([x0, x1, x2], [y0, y1, y2], "quadratic")
  # f = lambda x : 1 / g(x)
  #########

  # interp = bezier_3_points([x0, x1, x2], [y0, y1, y2])
  
  aa = ((x0 - x1) * (y1 - y2) - (x1 - x2) * (y0 - y1)) / (2 * (-x0 + x2) * (x0 - x1) * (x1 - x2))
  bb = (-x0 * (x0 - x1) * (y1 - y2) + x2 * (x1 - x2) * (y0 - y1)) / ((-x0 + x2) * (x0 - x1) * (x1 - x2))
  cc = -aa * x0**2 - bb * x0 + y0
  # print("abc:",aa, bb, cc)
  # print("xxx:",x0, x1, x2)
  # print("yyy:",y0, y1, y2)

  def g(x):
    x = np.array(x, dtype=np.float64)
    return (aa * x**2 + bb * x + cc)

  def f(x):
    x = np.array(x, dtype=np.float64)
    return 1 / (aa * x**2 + bb * x + cc)

  fig.add_trace(
    go.Scatter(
      name="curvatures",
      x=curvatures_X,
      y=curvatures_Y,
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
      name="curvature points",
      x=[x0, x1, x2],
      y=[y0, y1, y2],
      mode="markers",
      marker=dict(
          # size=8,
          color="pink",
          # showscale=False
      )
    )
  )

  XX = np.arange(n)
  np.savetxt("XX.csv", XX, delimiter=",")
  fig.add_trace(
    go.Scatter(
      name="curvatures",
      x=XX,
      y=g(XX),
      mode="lines",
      line=dict(
          # size=8,
          color="green",
          # showscale=False
      )
    )
  )

  return f


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
      mode="lines",
      line=dict(
          width=1,
          color="green"
      )
    )
  )
  return X, Y


def get_frontier(X, Y, nn):
  '''extracts the envelope via snowball method'''
  radius = get_curvature(X, Y, nn)
  n = len(X)
  idx1 = 0
  idx2 = 1
  envelope_X = [X[0]]
  envelope_Y = [Y[0]]
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
      envelope_X.append(X[idx2])
      envelope_Y.append(Y[idx2])
      idx1 = idx2
      idx2 += 1
  return envelope_X, envelope_Y



'''==============='''
''' Read wav file '''
'''==============='''
fig = go.Figure()

name = "piano33"
W, fps = read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
print(n)
X = np.arange(n)

# W = W [ : W.size // 10]

pulses = signal_to_pulses(W)

areas_X, areas_Y = get_pulses_area(pulses)

pos_pulses, neg_pulses, pos_noises, neg_noises = split_pulses(pulses)

pos_X, neg_X = np.array([p.x for p in pos_pulses]), np.array([p.x for p in neg_pulses])
pos_Y, neg_Y = np.array([p.y for p in pos_pulses]), np.array([p.y for p in neg_pulses])
pos_L, neg_L = np.array([p.len for p in pos_pulses]) , np.array([p.len for p in neg_pulses])

pos_frontier_X, pos_frontier_Y = get_frontier(pos_X, pos_Y, n)
neg_frontier_X, neg_frontier_Y = get_frontier(neg_X, neg_Y, n)

# pos_env_X, pos_env_Y = smooth(pos_frontier_X, pos_frontier_Y, n)
# neg_env_X, neg_env_Y = smooth(neg_frontier_X, neg_frontier_Y, n)

# XX, YY, II = get_knots_indices(pos_frontier_X, pos_frontier_Y, 2)

# A = constrained_least_squares_arbitrary_intervals(pos_frontier_X, pos_frontier_Y, II, k=3)
# YY = coefs_to_array_arbitrary_intervals(A, X, II, n)
# XX = X

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''


fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",
  # yaxis = dict(
  #   scaleanchor = "x",
  #   scaleratio = 1,
  # ),
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

''' Signal '''
fig.add_trace(
  go.Scatter(
    name="Signal",
    x=X,
    y=W,
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Normalized Signal",
#     x=np.arange(W.size),
#     y=W_normalized,
#     mode="lines",
#     line=dict(
#         # size=8,
#         color="orange",
#         # showscale=False
#     )
#   )
# )

''' Pulses '''
fig.add_trace(
  go.Scatter(
    name="Pulses",
    x=areas_X,
    y=areas_Y,
    fill="tozeroy",
    mode="none",
    fillcolor="rgba(0,0,0,0.16)",
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+ Amplitudes",
    x=pos_X,
    y=pos_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="markers",
    marker=dict(
        size=6,
        color="black",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="- Amplitudes",
    x=neg_X,
    y=neg_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="markers",
    marker=dict(
        size=6,
        color="black",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+ Frontier",
    x=pos_frontier_X,
    y=pos_frontier_Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="- Frontier",
    x=neg_frontier_X,
    y=neg_frontier_Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Knots",
#     x=np.array([[pos_frontier_X[i], pos_frontier_X[i], None] for i in II]).flat,
#     y=np.array([[-1, 1, None] for _ in II]).flat,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="Pos Envelope Avgs",
#     x=XX,
#     y=YY,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         # width=1,
#         color="blue",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

fig.show(config=dict({'scrollZoom': True}))

exit()

'''============================================================================'''
'''                                     FT                                     '''
'''============================================================================'''

FT = np.abs(np.fft.rfft(W))
FT = FT / np.max(FT)
FT_normalized = np.abs(np.fft.rfft(W_normalized))
FT_normalized = FT_normalized / np.max(FT_normalized)

fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",
  # yaxis = dict(
  #   scaleanchor = "x",
  #   scaleratio = 1,
  # ),
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
    name="Original FT",
    # x=pos_X,
    y=FT,
    # hovertext=np.arange(len(pos_pulses)),
    mode="lines",
    line=dict(
        # size=6,
        color="black",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="Normalized FT",
    # x=pos_X,
    y=FT_normalized,
    # hovertext=np.arange(len(pos_pulses)),
    mode="lines",
    line=dict(
        # size=6,
        color="red",
        # showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)


#########################################################
#########################################################
#########################################################

# rotate_ccw = np.array([[0, -1], [1, 0]])
# squares_X = []
# squares_Y = []
# for i in range(pos_X.size - 1):
#   x0, x1, y0, y1 = pos_X[i], pos_X[i + 1], pos_Y[i], pos_Y[i + 1]
#   x, y = rotate_ccw @ np.array([x1 - x0, y1 - y0])
#   squares_X.append(x0)     ; squares_Y.append(y0)
#   squares_X.append(x0 + x) ; squares_Y.append(y0 + y)
#   squares_X.append(x1 + x) ; squares_Y.append(y1 + y)
#   squares_X.append(x1)     ; squares_Y.append(y1)
#   squares_X.append(x0)     ; squares_Y.append(y0)
#   squares_X.append(None)   ; squares_Y.append(None)

# new_pos_pulses = []
# for i, p in enumerate(pos_pulses):
#   if pos_curvatures[i] >= avg_pos_curvature:
#     p.noise = True
#   else:
#     new_pos_pulses.append(p)

# srtd = np.argsort([p.y for p in pulses])
# sorted_pulses = [pulses[i] for i in srtd]
# X = [p.x for p in sorted_pulses]
# Y = [p.y for p in sorted_pulses]
