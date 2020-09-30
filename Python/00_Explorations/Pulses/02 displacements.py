import plotly.graph_objects as go
# import alphashape
import numpy as np
from Helper import read_wav
# from scipy.signal import savgol_filter
import numpy.polynomial.polynomial as poly
from scipy.spatial import Delaunay

class Pulse:
  def __init__(self, x0, W, is_noise = False):
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
  ''' Return pulses X & Y area vectors '''
  X, Y = [], []
  for p in pulses:
    X.append(p.start - .5); Y.append(0)
    X.append(p.start - .5); Y.append(p.y)
    X.append(p.end - .5)  ; Y.append(p.y)
    X.append(p.end - .5)  ; Y.append(0)
    X.append(None)        ; Y.append(None)
  return X, Y


def get_pulses_curvatures_poly(X, Y):
  '''Return max curvatures, avg and r for the poly approximation of 3 pulses amplitude at a time'''
  # pulses = [pulses[0]] + pulses + [pulses[-1]] # FIXME

  curvatures = []
  for i in range(len(X) - 2):
    x0, x1, x2 = X[i], X[i + 1], X[i + 2]
    y0, y1, y2 = Y[i], Y[i + 1], Y[i + 2]
    x, y = np.array([x0, x1, x2]), np.array([y0, y1, y2])
    # y = y / y1
    a0, a1, a2 = poly.polyfit(x, y, 2)
    # c = np.abs(2 * a2) # max curvature
    # c = ((2 * a2 * x1 + a1) / np.sqrt((2 * a2 * x1 + a1)**2 + 1) - (2 * a2 * x0 + a1) / np.sqrt((2 * a2 * x0 + a1)**2 + 1)) / (x1-x0)
    c = 2 * a2 / ((a1 + 2*a2*x)**2 + 1)**(3/2)
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


def constrained_least_squares(X, Y, q=4, k=4):
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


def coefs_to_array(A, X): # create array of size 'n' from a coefficient matrix 'A'
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

'''==============='''
''' Read wav file '''
'''==============='''

name = "piano33"
W, fps = read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude

# W = W [ : W.size // 10]

pulses = signal_to_pulses(W)

# avg_len = np.average([p.len for p in pulses])
# avg_y = np.average([abs(p.y) for p in pulses])
# for p in pulses:
#   p.len = p.len / avg_len
#   p.y = p.y / avg_y

areas_X, areas_Y = get_pulses_area(pulses)

pos_pulses, neg_pulses, pos_noises, neg_noises = split_pulses(pulses)

# pos_curvatures, avg_pos_curvature, radius = get_pulses_curvatures_poly(pos_pulses)

pos_X, neg_X = np.array([p.x for p in pos_pulses]), np.array([p.x for p in neg_pulses])
pos_Y, neg_Y = np.array([p.y for p in pos_pulses]), np.array([p.y for p in neg_pulses])
pos_L, neg_L = np.array([p.len for p in pos_pulses]) , np.array([p.len for p in neg_pulses])

dis_X = []
dis_Y = []
for i in range(pos_X.size - 1):
  x  = (pos_X[i + 1] + pos_X[i]) / 2
  dy = abs(pos_Y[i + 1] - pos_Y[i])
  dis_X.append(x)
  dis_Y.append(dy)


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
    x=np.arange(W.size),
    y=W,
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    )
  )
)

''' Pulses info '''
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
    name="Positive Amplitudes",
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
    name="Negative Amplitudes",
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

# fig.add_trace(
#   go.Scatter(
#     name="Pos Tri",
#     x=pos_X_tri,
#     y=pos_Y_tri,
#     mode="lines",
#     line=dict(
#         width=.5,
#         color="red",
#         # showscale=False
#     )
#   )
# )


fig.show(config=dict({'scrollZoom': True}))

# exit()

'''============================================================================'''
'''                                    ENVE                                    '''
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

fig.add_trace(
  go.Scatter(
    name="Positive Amplitudes",
    x=pos_X,
    y=pos_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="lines+markers",
    marker=dict(
        size=6,
        color="black",
        # showscale=False
    )
  )
)


fig.add_trace(
  go.Scatter(
    name="lms",
    x=dis_X,
    y=dis_Y,
    # fill="toself",
    mode="lines+markers",
    marker=dict(
        # width=1,
        color="red",
        # showscale=False
    )
  )
)


# for x, y in zip([0], [0]):
#   fig.add_shape(
#     type="circle",
#     xref="x",
#     yref="y",
#     x0= x - radius,
#     y0= y - 2 * radius,
#     x1= x + radius,
#     y1= y,
#     line_color="LightSeaGreen",
#   )


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
