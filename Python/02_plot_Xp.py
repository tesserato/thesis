import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
# import numpy.polynomial.polynomial as poly
# from collections import Counter
# from math import gcd
import hp as hp
# from scipy.signal import savgol_filter
# from plotly.subplots import make_subplots


'''==============='''
''' Read wav file '''
'''==============='''


name = "alto"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos_orig, Xneg_orig = se.get_frontiers(W)

Xpos = hp.refine_frontier_iter(Xpos_orig, W)
# while not Xposnew is None:
#   Xpos = np.unique(np.hstack([Xpos, Xposnew])).astype(np.int)
#   Xposnew = hp.refine_frontier(Xpos, W)

Xneg = hp.refine_frontier_iter(Xneg_orig, W)
# while not Xnegnew is None:
  # Xneg = np.unique(np.hstack([Xneg, Xnegnew])).astype(np.int)
  # Xnegnew = hp.refine_frontier(Xneg, W)




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
    y=Xpos_orig,
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
    y=Xneg_orig,
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
    y=Xpos,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Refined Minima", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xneg,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)


fig.show(config=dict({'scrollZoom': True}))