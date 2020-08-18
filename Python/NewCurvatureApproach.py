import plotly.graph_objects as go
import numpy as np
from scipy.interpolate.interpolate import interp1d
from Helper import draw_circle, read_wav, get_pulses_area, split_pulses, get_circle, Pulse#, save_wav, get_frontier
import numpy.polynomial.polynomial as poly


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

  fig.add_trace(
    go.Scatter(
      name="Circles",
      legendgroup="Circles",
      x=[0],
      y=[0],
      mode="none",
      visible = "legendonly",
      line=dict(
          width=1,
          color="green"
      )
    )
  )

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

      draw_circle(xc, yc, r, fig)

      Frontier.append(Pulses[idx2])
      idx1 = idx2
      idx2 += 1
  Frontier.append(Pulses[-1])
  # f = interp1d(frontier_X, frontier_Y, kind="linear")
  # X = np.arange(nn)
  return [p.x for p in Frontier], [p.y for p in Frontier]


'''==============='''
''' Read wav file '''
'''==============='''
fig = go.Figure()

name = "brass"
W, fps = read_wav(f"Samples/{name}.wav")

# W = W [:10000]

W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
print(n)
X = np.arange(n)


pulses = signal_to_pulses(W)

areas_X, areas_Y = get_pulses_area(pulses)

pos_pulses, neg_pulses, pos_noises, neg_noises = split_pulses(pulses)

pos_X, neg_X = np.array([p.x for p in pos_pulses]), np.array([p.x for p in neg_pulses])
pos_Y, neg_Y = np.array([p.y for p in pos_pulses]), np.array([p.y for p in neg_pulses])
pos_L, neg_L = np.array([p.len for p in pos_pulses]) , np.array([p.len for p in neg_pulses])

'''pos frontier'''
scaling = np.average(pos_L) / np.average(pos_Y)

pos_Y = pos_Y * scaling
for p in pos_pulses:
  p.y = p.y * scaling

pos_frontier_X, pos_frontier_Y = get_frontier(pos_pulses)
_, curvature_X, curvature_Y, smooth_curvature_X, smooth_curvature_Y = get_curvature_function(pos_X, pos_Y)
pos_frontier_Y = pos_frontier_Y / scaling
pos_Y = pos_Y / scaling

'''neg frontier'''
scaling = np.average(neg_L) / np.average(neg_Y)

neg_Y = neg_Y * scaling
for p in neg_pulses:
  p.y = p.y * scaling

neg_frontier_X, neg_frontier_Y = get_frontier(neg_pulses)
_, curvature_X, curvature_Y, smooth_curvature_X, smooth_curvature_Y = get_curvature_function(neg_X, -np.array(neg_Y))
neg_frontier_Y = np.array(neg_frontier_Y) / scaling
neg_Y = neg_Y / scaling

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''


fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",

  # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
  # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

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
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
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

fig.add_trace(
  go.Scatter(
    name="Pulses", # <|<|<|<|<|<|<|<|<|<|<|<|
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
    name="+Amps", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=pos_X,
    y=pos_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="lines+markers",
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
    name="-Amps", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=neg_X,
    y=neg_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="lines+markers",
    marker=dict(
        size=6,
        color="black",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="+Curvatures", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=curvature_X,
#     y=curvature_Y,
#     # hovertext=np.arange(len(pos_pulses)),
#     mode="markers",
#     marker=dict(
#         # size=6,
#         color="green",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

fig.add_trace(
  go.Scatter(
    name="+ Smooth Curvatures", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=smooth_curvature_X,
    y=smooth_curvature_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="lines",
    line=dict(
        # size=6,
        color="blue",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
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
    name="-Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
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

fig.add_trace(
  go.Scatter(
    name="avg vector", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, np.average(pos_X[1:]-pos_X[:-1])],
    y=[0, np.average(pos_Y[1:]-pos_Y[:-1])],
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

fig.show(config=dict({'scrollZoom': True}))