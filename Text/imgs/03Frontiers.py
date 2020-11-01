import sys
# sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
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

parent_path = str(Path(os.path.abspath('./')).parents[0])
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


FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="<b><i>i</i></b>",
  yaxis_title="<b>Amplitude</b>",
  paper_bgcolor='rgba(0,0,0,0)',
  plot_bgcolor='rgba(0,0,0,0)',
  legend=dict(orientation='h', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT
fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=2, zerolinecolor='gray')


fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(
        width=.8,
        color="gray",
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
    name="Frontiers", # <|<|<|<|<|<|<|<|<|<|<|<|
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

# fig.show()
save_name = "./imgs/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=605, height=400, engine="kaleido", format="svg")
print("saved:", save_name)

