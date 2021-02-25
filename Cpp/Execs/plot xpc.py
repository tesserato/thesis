from os import linesep
from statistics import mode
import numpy as np
import plotly.graph_objects as go
import signal_envelope as se
from scipy import interpolate


def get_pcs(Xpc, W):
  Xpc = Xpc.astype(int)
  # amp = np.max(np.abs(W))
  # max_T = int(np.max(np.abs(Xpc[1:] - Xpc[:-1])))
  Xlocal = np.linspace(0, 1, mode(Xpc[1:] - Xpc[:-1]))

  orig_pcs = []
  norm_pcs = []
  for i in range(2, Xpc.size):
    x0 = Xpc[i - 1]
    x1 = Xpc[i] + 1
    orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1-x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      # Ylocal = Ylocal / np.max(np.abs(Ylocal)) * amp
      norm_pcs.append(Ylocal)
  return np.average(np.array(norm_pcs), 0), orig_pcs, norm_pcs

def to_plot(Matrix):
  X = []
  Y = []
  for line in Matrix:
    for x, y in enumerate(line):
      X.append(x)
      Y.append(y)
    X.append(None)
    Y.append(None)
  return X, Y

name = "alto"

'''###### Read wav file ######'''
W, fps = se.read_wav(f"{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
W = W / amp
n = W.size

'''###### Read Pseudo-cycles info ######'''
Xpc = np.genfromtxt(name + ".csv", delimiter=",")

X_xpcs, Y_xpcs = [], []
for x in Xpc:
  X_xpcs.append(x)
  X_xpcs.append(x)
  X_xpcs.append(None)
  Y_xpcs.append(-1)
  Y_xpcs.append(1)
  Y_xpcs.append(None)


average_waveform, orig_waveforms, norm_waveforms = get_pcs(Xpc, W)

# norm_waveforms = norm_waveforms - average_waveform



'''============================================================================'''
'''                                 PLOT SIGNAL                                '''
'''============================================================================'''
fig = go.Figure()
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
  go.Scattergl(
    name="W", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=XX,
    y=W,
    # fill="toself",
    mode="lines+markers",
    line=dict(
        width=1,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="Pseudo-cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X_xpcs,
    y=Y_xpcs,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))

'''============================================================================'''
'''                               PLOT WAVEFORM                                '''
'''============================================================================'''
fig = go.Figure()
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

X, Y = to_plot(norm_waveforms)
fig.add_trace(
  go.Scattergl(
    name="Normalized", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="rgba(38, 12, 12, 0.2)",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="Average", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=XX,
    y=average_waveform,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))

