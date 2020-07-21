import plotly.graph_objects as go
from scipy.interpolate import interp1d, UnivariateSpline
import numpy as np
from Helper import read_wav
from scipy.signal import savgol_filter
import numpy.polynomial.polynomial as poly

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


def get_average_curvature(X, Y):
  '''Return average curvature and radius of the equivalent circle for the poly approximation of 4 points amplitude at a time'''
  curvatures = []
  for i in range(len(X) - 3):
    x0, x1, x2, x3 = X[i], X[i + 1], X[i + 2], X[i + 3]
    y0, y1, y2, y3 = Y[i], Y[i + 1], Y[i + 2], Y[i + 3]
    x, y = np.array([x0, x1, x2, x3]), np.array([y0, y1, y2, y3])
    a0, a1, a2 = poly.polyfit(x, y, 2)
    c = 2 * a2 / ((a1 + a2 * (x1 + x2))**2 + 1)**(3/2)
    curvatures.append(c)
  average_curvature = np.abs(np.average(curvatures))
  radius = 1 / average_curvature
  return  average_curvature, radius


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
        U[i2, i1 * k + i3] = X[i2]**i3 #f"X_p[{i2}]**{i3}"

  '''populating V'''
  for i in range(q - 1):
    for j in range(k):
      V[2 * i, i * k + j] = X[(i + 1) * l]**j #f"X_p[({i + 1}) * l]**{j}"
      V[2 * i, i * k + k + j] = -X[(i + 1) * l]**j #f"-X_p[({i + 1}) * l]**{j}"
  for i in range(q - 1):
    for j in range(1, k):
      V[2 * i + 1, i * k + j] = j * X[(i + 1) * l] ** (j - 1) #f"{j} * X_p[({i + 1}) * l]**({j-1})"
      V[2 * i + 1, i * k + k + j] = -j * X[(i + 1) * l] ** (j - 1) #f"{-j} * X_p[({i + 1}) * l]**({j-1})"

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


def get_circle(x0, y0, x1, y1, r):
  '''returns center of circle that passes through two points'''
  radsq = r * r
  q = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

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


def get_frontier(X, Y):
  '''extracts the envelope via rolling circle method'''
  _, r = get_average_curvature(X, Y)
  n = len(X)
  idx1 = 0
  idx2 = 1
  envelope_X = [X[0]]
  envelope_Y = [Y[0]]
  while idx2 < n:
    xc, yc = get_circle(X[idx1], Y[idx1], X[idx2], Y[idx2], r)
    empty = True
    for i in range(idx2 + 1, n):
      if np.sqrt((xc - X[i])**2 + (yc - Y[i])**2) < r:
        empty = False
        idx2 += 1
        break
    if empty:
      # envelope_X.append(X[idx1]) , envelope_Y.append(Y[idx1])
      envelope_X.append(X[idx2]) , envelope_Y.append(Y[idx2])
      idx1 = idx2
      idx2 += 1
  return envelope_X, envelope_Y


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
  Yn = np.zeros(len(Y))

  '''first value'''
  s = Y[0] + Y[1] + Y[2]
  h0, h1 = Y[0] / s, Y[1] / s

  d = X[2] - X[0]
  d0, d1 = 1, 1 - (X[1] - X[0]) / d

  Yn[0] = (Y[0] * h0 * d0 + Y[1] * h1 * d1) / (h0 * d0 + h1 * d1)

  '''last value''' 
  s = Y[-2] + Y[-1]
  h0, h1 = Y[-2] / s, Y[-1] / s

  d = X[-1] - X[-3]
  d0, d1 = 1, 1 - (X[-1] - X[-2]) / d

  Yn[-1] = (Y[-2] * h0 * d0 + Y[-1] * h1 * d1) / (h0 * d0 + h1 * d1)
  
  for i in range(1, len(Y) - 1):
    y0, y, y1 = Y[i - 1], Y[i], Y[i + 1]

    d = X[i + 1] - X[i - 1]
    d0 = 1 - (X[i] - X[i - 1]) / d
    d1 = 1 - (X[i + 1] - X[i]) / d

    s = y0 + y + y1
    h = y / s
    h0 = y0 / s
    h1 = y1 / s

    Yn[i] = (y * h + d0 * y0 * h0 + d1 * y1 * h1) / (h + h0 * d0 + h1 * d1)

  f = UnivariateSpline(X, Yn, s=0)
  return f(np.arange(n))


'''==============='''
''' Read wav file '''
'''==============='''

name = "tom"
W, fps = read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
X = np.arange(n)

# W = W [ : W.size // 10]

pulses = signal_to_pulses(W)

areas_X, areas_Y = get_pulses_area(pulses)

pos_pulses, neg_pulses, pos_noises, neg_noises = split_pulses(pulses)

pos_X, neg_X = np.array([p.x for p in pos_pulses]), np.array([p.x for p in neg_pulses])
pos_Y, neg_Y = np.array([p.y for p in pos_pulses]), np.array([p.y for p in neg_pulses])
pos_L, neg_L = np.array([p.len for p in pos_pulses]) , np.array([p.len for p in neg_pulses])

pos_env_X, pos_env_Y = get_frontier(pos_X, pos_Y)

neg_env_X, neg_env_Y = get_frontier(neg_X, neg_Y)


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
    )
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
    fillcolor="rgba(0,0,0,0.16)"
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
    )
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
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="+ Frontier",
    x=pos_env_X,
    y=pos_env_Y,
    # fill="toself",
    mode="lines+markers",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="- Frontier",
    x=neg_env_X,
    y=neg_env_Y,
    # fill="toself",
    mode="lines+markers",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    )
  )
)

YY = smooth(pos_env_X, pos_env_Y, n)


fig.add_trace(
  go.Scatter(
    name="YY",
    x=X,
    y=YY,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        # showscale=False
    )
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Ye0",
#     x=Xs,
#     y=Ye0,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="blue",
#         # showscale=False
#     )
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="Ye1",
#     x=Xs,
#     y=Ye1,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="blue",
#         # showscale=False
#     )
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="Yo0",
#     x=Xs,
#     y=Yo0,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="blue",
#         # showscale=False
#     )
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="Yo1",
#     x=Xs,
#     y=Yo1,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="blue",
#         # showscale=False
#     )
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="Negative Envelope Cubic",
#     x=X,
#     y=savgol_filter(f_neg(X), 101, 3),
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="blue",
#         # showscale=False
#     )
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
