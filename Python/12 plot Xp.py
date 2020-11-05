import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import numpy.polynomial.polynomial as poly
from collections import Counter
from math import gcd
import hp as hp
from scipy.signal import savgol_filter
from plotly.subplots import make_subplots


'''==============='''
''' Read wav file '''
'''==============='''


name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)

Xposadicional = hp.refine_frontier(Xpos, W)
Xposref = np.sort(np.hstack([Xpos, Xposadicional])).astype(np.int)

Xnegadicional = hp.refine_frontier(Xneg, W)
Xnegref = np.sort(np.hstack([Xneg, Xnegadicional])).astype(np.int)



'''============================================================================'''
'''                              PLOT LINES                                    '''
'''============================================================================'''


fig = fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="Length",
  yaxis_title="Number of Ocurrences",

  # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
  # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=12
  )
)

fig.add_trace(
  go.Scatter(
    name="Maxima", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xpos,
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
    name="Minima", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xneg,
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
    name="Refined Maxima", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xposref,
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
    name="Refined Minima", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xnegref,
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


fig.show(config=dict({'scrollZoom': True}))