import plotly.graph_objects as go
# import random
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


'''==============='''
''' Read wav file '''
'''==============='''

name = "alto"
W, fps = read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude

# W = W [ : W.size // 100]

# n = W.shape[0]
# X = np.arange(n)

pulses = signal_to_pulses(W)

areas_X, areas_Y = get_pulses_area(pulses)

pos_pulses, neg_pulses, pos_noises, neg_noises = split_pulses(pulses)

pos_X, neg_X = np.array([p.x for p in pos_pulses])   , np.array([p.x for p in neg_pulses])
pos_Y, neg_Y = np.array([p.y for p in pos_pulses])   , np.array([p.y for p in neg_pulses])
pos_L, neg_L = np.array([p.len for p in pos_pulses]) , np.array([p.len for p in neg_pulses])

points = np.array([[x, y] for x, y in zip(pos_X, pos_Y)])

tri = Delaunay(points)

# A = constrained_least_squares(pos_X, pos_Y, q=4, k=3)
# pos_Y_avg = coefs_to_array(A, pos_X)



# print(f"Max length of a pulse={max_length}")
# print(f"{len(pos_pulses) + len(neg_pulses)} of {len(pulses)} are valid (lenght > 2)")

# pos_Y_dev = pos_Y - pos_Y_avg

# fig = go.Figure()
# fig.add_trace(go.Histogram(x=pos_Y_dev, histnorm='probability'))
# fig.show()

# pos_Y_var = np.var(())
# pos_Y_dist = (pos_Y - pos_Y_avg) / pos_Y_var


# bins = 20
# mi = int(min(pos_Y_dist) - 1)
# ma = int(max(pos_Y_dist) + 1)
# print(mi, ma)
# freqs = np.zeros(ma - mi)
# XX = np.linspace(mi, ma, ma - mi)


# for v in pos_Y_dist:
#   freqs[int(round(v - mi))] += 1


'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white"
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
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
        size=3,
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
        size=3,
        color="black",
        # showscale=False
    )
  )
)

Xt = []
Yt = []
for triangle in points[tri.simplices]:
  # print(triangle,"\n")
  for point in triangle:
    x, y = point
    Xt.append(x)
    Yt.append(y)
  Xt.append(None)
  Yt.append(None)

fig.add_trace(
  go.Scatter(
    name="Tri",
    x=Xt,
    y=Yt,
    # hovertext=np.arange(len(pos_pulses)),    
    mode="lines",
    line=dict(
        width=.5,
        color="red",
        # showscale=False
    )
  )
)

# fig.show(config=dict({'scrollZoom': True}))


# fig = go.Figure()
# fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
# fig.update_layout(
#   # title = name,
#   xaxis_title="x",
#   yaxis_title="Amplitude",
#   # yaxis = dict(
#   #   scaleanchor = "x",
#   #   scaleratio = 1,
#   # ),
#   legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
#   margin=dict(l=5, r=5, b=5, t=5),
#   font=dict(
#   family="Computer Modern",
#   color="black",
#   size=18
#   )
# )

# ''' Signal '''
# fig.add_trace(
#   go.Scatter(
#     name="Signal",
#     x=XX,
#     y=freqs,
#     mode="lines",
#     line=dict(
#         # size=8,
#         color="silver",
#         # showscale=False
#     )
#   )
# )

fig.show(config=dict({'scrollZoom': True}))
exit()


fig.add_trace(
  go.Scatter(
    name="Positive Lengths",
    x=pos_pulses_X,
    y=pos_pulses_len,
    # hovertext=np.arange(len(pos_pulses)),
    mode="markers",
    marker=dict(
        size=3,
        color="red",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="Negative Lengths",
    x=neg_pulses_X,
    y=-neg_pulses_len,
    # hovertext=np.arange(len(neg_pulses)),
    mode="markers",
    marker=dict(
        size=3,
        color="red",
        # showscale=False
    )
  )
)


'''===================='''
''' Average amplitudes '''
'''===================='''

terms = 4
pos_args = poly.polyfit(pos_pulses_X, pos_pulses_Y, terms)
neg_args = poly.polyfit(neg_pulses_X, neg_pulses_Y, terms)
pos_pulses_Y_avg = poly.polyval(pos_pulses_X, pos_args)
neg_pulses_Y_avg = poly.polyval(neg_pulses_X, neg_args)

fig.add_trace(
  go.Scatter(
    x=pos_pulses_X,
    y=pos_pulses_Y_avg,
    # hovertext=np.arange(len(pos_pulses)),
    name="Positive Average Amplitude",
    mode="lines",
    line=dict(
        width=2,
        color="gray",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    x=neg_pulses_X,
    y=neg_pulses_Y_avg,
    # hovertext=np.arange(len(pos_pulses)),
    name="Negative Average Amplitude",
    mode="lines",
    line=dict(
        width=2,
        color="gray",
        # showscale=False
    )
  )
)

'''================='''
''' Average lengths '''
'''================='''

terms = 4
pos_args_len = poly.polyfit(pos_pulses_X, pos_pulses_len, terms)
neg_args_len = poly.polyfit(neg_pulses_X, neg_pulses_len, terms)
pos_pulses_len_avg = poly.polyval(pos_pulses_X, pos_args_len)
neg_pulses_len_avg = poly.polyval(neg_pulses_X, neg_args_len)

fig.add_trace(
  go.Scatter(
    x=pos_pulses_X,
    y=pos_pulses_len_avg,
    # hovertext=np.arange(len(pos_pulses)),
    name="Positive Average Lengths",
    mode="lines",
    line=dict(
        width=2,
        color="red",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    x=neg_pulses_X,
    y=-neg_pulses_len_avg,
    # hovertext=np.arange(len(pos_pulses)),
    name="Negative Average Lengths",
    mode="lines",
    line=dict(
        width=2,
        color="red",
        # showscale=False
    )
  )
)


fig.show(config=dict({'scrollZoom': True}))
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)

'''==============='''
''' Distributions '''
'''==============='''

original_dist = pos_pulses_Y_avg - pos_pulses_Y

# bins = 10
# values = np.round(() * bins + bins / 2)
# freqs = np.zeros(bins)

# for v in values:
#   freqs[int(v)] += 1

fig = go.Figure()
fig.layout.template ="plotly_white"
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
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
    name="Signal",
    y=np.random.rand(original_dist.size),
    x=original_dist,
    mode="markers",
    marker=dict(
        # size=8,
        color="silver",
        # showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))


# radius = .1
# for x, y in zip(pos_pulses_X, pos_pulses_Y):
#   fig.add_shape(
#     type="circle",
#     xref="x",
#     yref="y",
#     x0=x - radius,
#     y0=y - radius,
#     x1=x + radius,
#     y1=y + radius,
#     line_color="LightSeaGreen",
#   )


# # plot positive and negative averages
# terms = 4
# pos_args = poly.polyfit(X_pos_pulses, Y_pos_pulses, terms)
# neg_args = poly.polyfit(X_neg_pulses, Y_neg_pulses, terms)

# X_pulses = np.array([p.start + p.x for p in pulses])
# fitted_Y_pos_pulses = poly.polyval(X_pos_pulses, pos_args)
# fitted_Y_neg_pulses = poly.polyval(X_neg_pulses, neg_args)

# fig.add_trace(
#   go.Scatter(
#     x=X_pos_pulses,
#     y=fitted_Y_pos_pulses,
#     # hovertext=np.arange(ratios.size),
#     name="Positive Average",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # dash="dash" # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
#         # showscale=False
#     )
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     x=X_neg_pulses,
#     y=fitted_Y_neg_pulses,
#     # hovertext=np.arange(ratios.size),
#     name="Negative Average",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # dash="dash"
#         # showscale=False
#     )
#   )
# )

# pos_std = []
# for a, b in zip(Y_pos_pulses, fitted_Y_pos_pulses):
#   aa = abs(a)
#   bb = abs(b)
#   pos_std.append(min(aa, bb) / max(aa, bb))

# pos_std = np.average(pos_std)

# fig.add_trace(
#   go.Scatter(
#     x=X_pos_pulses,
#     y=fitted_Y_pos_pulses - fitted_Y_pos_pulses * pos_std,
#     # hovertext=np.arange(ratios.size),
#     name="Positive std",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         dash="dash" # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
#         # showscale=False
#     )
#   )
# )

# neg_std = []
# for a, b in zip(Y_neg_pulses, fitted_Y_neg_pulses):
#   aa = abs(a)
#   bb = abs(b)
#   neg_std.append(min(aa, bb) / max(aa, bb))

# neg_std = np.average(neg_std)


# fig.add_trace(
#   go.Scatter(
#     x=X_neg_pulses,
#     y=fitted_Y_neg_pulses - fitted_Y_neg_pulses * neg_std,
#     # hovertext=np.arange(ratios.size),
#     name="Negative std",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         dash="dash" # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
#         # showscale=False
#     )
#   )
# )

# fig.show(config=dict({'scrollZoom': True}))
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)
