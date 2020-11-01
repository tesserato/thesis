import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import numpy.polynomial.polynomial as poly
from collections import Counter
from math import gcd
import hp


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



'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
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
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xpos,
    y=W[Xpos],
    # fill="toself",
    mode="markers",
    line=dict(
        # width=1,
        color="red",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

XX = []
YY = []
for x in hp.linearize_pc(Xpos):
  XX.append(x), XX.append(x), XX.append(None)
  YY.append(-amp), YY.append(amp), YY.append(None)

fig.add_trace(
  go.Scatter(
    name="+Frontier sm", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=XX,
    y=YY,
    # fill="toself",
    mode="lines",
    line=dict(
        # width=1,
        color="blue",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="-Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xneg,
    y=W[Xneg],
    # fill="toself",
    mode="markers",
    line=dict(
        # width=1,
        color="red",
        # showscale=False
    ),
    visible = "legendonly"
  )
)


fig.show(config=dict({'scrollZoom': True}))