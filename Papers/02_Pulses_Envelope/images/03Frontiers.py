import sys
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
import os
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate.interpolate import interp1d
from Helper import read_wav, get_pulses_area, split_pulses, signal_to_pulses, get_frontier#, save_wav, get_frontier
from scipy.signal import savgol_filter

# os.path.dirname(os.path.dirname(path))

'''==============='''
''' Read wav file '''
'''==============='''
fig = go.Figure()

parent_path = str(Path(os.path.abspath('./')).parents[1])
print(parent_path)
name = "piano33"
W, fps = read_wav(f"{parent_path}/Python/Samples/{name}.wav")

W = W[100103 : 100775]

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

f_pos = interp1d(pos_frontier_X, pos_frontier_Y, fill_value="extrapolate", kind="linear")
f_neg = interp1d(neg_frontier_X, neg_frontier_Y, fill_value="extrapolate", kind="linear")
XX = np.linspace(0, n, 4 * n)
envelope = (f_pos(XX) - f_neg(XX)) / 2
envelope = savgol_filter(envelope, 101, 3)


'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''


fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="$i$",
  yaxis_title="Amplitude",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)
fig.update_xaxes(showline=False, showgrid=False, zeroline=False)#, range=[100103, 100773])
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=2, zerolinecolor='gray')#, range=[-.1, .1])


fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(
        width=.8,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
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
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+Amps", # <|<|<|<|<|<|<|<|<|<|<|<|
    showlegend=False,
    x=pos_X,
    y=pos_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="markers",
    marker=dict(
        size=6,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="-Amps", # <|<|<|<|<|<|<|<|<|<|<|<|
    showlegend=False,
    x=neg_X,
    y=neg_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="markers",
    marker=dict(
        size=6,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=pos_frontier_X,
    y=pos_frontier_Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Negative Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    showlegend=False,
    x=neg_frontier_X,
    y=neg_frontier_Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Envelope", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=envelope,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="silver",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)
fig.add_trace(
  go.Scatter(
    name="Envelope", # <|<|<|<|<|<|<|<|<|<|<|<|
    showlegend=False,
    x=X,
    y=-envelope,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="silver",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Pseudo-Cycles",
    x=[167.5, 167.5, None, 419.5, 419.5],
    y=[-1, 1, None, -1, 1],
    # marker_symbol="line-ns",
    mode="lines",
    line=dict(
      # width=5,
      color="black",
      dash="dash"
    )
    # marker_line_width=2, 
    # marker_size=50,
  )
)

fig.show(config=dict({'scrollZoom': True}))
fig.write_image("./03Frontiers.pdf", width=800, height=400, scale=1)
