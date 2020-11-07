import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
# import numpy.polynomial.polynomial as poly
# from collections import Counter
from statistics import mode
# from math import gcd
import hp as hp
# from scipy.signal import savgol_filter
# from plotly.subplots import make_subplots


'''==============='''
''' Read wav file '''
'''==============='''

def stats(L):
  avg = np.average(L)
  mde = mode(L)
  std_avg = np.average(np.abs(L - avg))
  std_mde = np.average(np.abs(L - mde))
  print("std:", std_avg, std_mde, np.min(L), mde, np.max(L))
  return avg, std_avg, mde, std_mde, np.std(L)


name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)
posL = Xpos[1:] - Xpos[:-1]
pos_avg, pos_std_avg, pos_mde, pos_std_mde, pos_std = stats(posL.astype(np.int))

negL = Xneg[1:] - Xneg[:-1]


Xpos = hp.refine_frontier_iter(Xpos, W)
Xneg = hp.refine_frontier_iter(Xneg, W)
posL_ref = Xpos[1:] - Xpos[:-1]
negL_ref = Xneg[1:] - Xneg[:-1]

'''============================================================================'''
'''                              PLOT LINES                                    '''
'''============================================================================'''


fig = fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="T",
  yaxis_title="Length (frames)",

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
    name="Positive Periods", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=posL,
    # fill="toself",
    mode="markers",
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
    name="Positive Periods Average", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, posL.size],
    y=[pos_avg, pos_avg],
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
    name="Positive Periods avg std", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, posL.size, None, 0, posL.size],
    y=[pos_avg + pos_std_avg, pos_avg + pos_std_avg, None, pos_avg - pos_std_avg, pos_avg - pos_std_avg],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Positive Periods Mode", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, posL.size],
    y=[pos_mde, pos_mde],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="blue",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Positive Periods mde std", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, posL.size, None, 0, posL.size],
    y=[pos_mde + pos_std_mde, pos_mde + pos_std_mde, None, pos_mde - pos_std_mde, pos_mde - pos_std_mde],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="blue",
        dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Negative Periods", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=negL,
    # fill="toself",
    mode="markers",
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
    name="Refined Positive Periods", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=posL_ref,
    # fill="toself",
    mode="markers",
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
    name="Refined Negative Periods", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=negL_ref,
    # fill="toself",
    mode="markers",
    line=dict(
        width=1,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)


fig.show(config=dict({'scrollZoom': True}))