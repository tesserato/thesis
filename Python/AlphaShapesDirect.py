import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
import numpy.polynomial.polynomial as poly
import wave
from scipy.signal import savgol_filter
from Wheel import frontiers

def read_wav(path): 
  """returns signal & fps"""
  wav = wave.open(path , 'r')
  signal = np.frombuffer(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps




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




pos_frontierX, pos_frontierY, neg_frontierX, neg_frontierY = frontiers(W)

frontierX = np.concatenate([pos_frontierX, neg_frontierX])
frontierY = np.concatenate([pos_frontierY, np.abs(neg_frontierY)])

idxs = np.argsort(frontierX)
frontierX = frontierX[idxs]
frontierY = frontierY[idxs]



# smooth_frontierY = savgol_filter(frontierY, n // frontierY.size + 1, 2)

# f = interp1d(frontierX, smooth_frontierY, kind="quadratic", fill_value="extrapolate", assume_sorted=False)

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

# fig.add_trace(
#   go.Scatter(
#     name="Flat Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=X,
#     y=W / f(X),
#     mode="lines",
#     line=dict(
#         # size=8,
#         color="gray",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

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

# fig.add_trace(
#   go.Scatter(
#     name="+Amps", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=posX,
#     y=posY,
#     # hovertext=np.arange(len(pos_pulses)),
#     mode="lines+markers",
#     marker=dict(
#         size=6,
#         color="black",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="-Amps", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=negX,
#     y=negY,
#     # hovertext=np.arange(len(pos_pulses)),
#     mode="lines+markers",
#     marker=dict(
#         size=6,
#         color="black",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

fig.add_trace(
  go.Scatter(
    name="+Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=pos_frontierX,
    y=pos_frontierY,
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
    x=neg_frontierX,
    y=neg_frontierY,
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
    name="Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=frontierX,
    y=frontierY,
    # fill="toself",
    mode="markers+lines",
    line=dict(
        width=1,
        color="blue",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Smooth Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=frontierX,
#     y=smooth_frontierY,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="green",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )


fig.show(config=dict({'scrollZoom': True}))