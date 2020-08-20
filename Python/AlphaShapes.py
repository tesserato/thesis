import plotly.graph_objects as go
import numpy as np
from scipy.interpolate.interpolate import interp1d
from Helper import draw_circle, read_wav, get_pulses_area, split_pulses, get_circle, signal_to_pulses, get_curvature_function, get_frontier#, save_wav, get_frontier
import numpy.polynomial.polynomial as poly


'''==============='''
''' Read wav file '''
'''==============='''
fig = go.Figure()

name = "piano33"
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

for p in pos_pulses:
  p.y = p.y * scaling

pos_frontier_X, pos_frontier_Y = get_frontier(pos_pulses)
pos_frontier_Y = pos_frontier_Y / scaling

'''neg frontier'''
scaling = np.average(neg_L) / np.average(neg_Y)

for p in neg_pulses:
  p.y = p.y * scaling

neg_frontier_X, neg_frontier_Y = get_frontier(neg_pulses)
neg_frontier_Y = np.array(neg_frontier_Y) / scaling

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