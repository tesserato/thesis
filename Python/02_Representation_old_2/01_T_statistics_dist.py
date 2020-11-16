import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
from statistics import mode
import hp as hp

'''==============='''
''' Read wav file '''
'''==============='''

# def stats(L):
#   avg = np.average(L)
#   mde = mode(L)
#   std_avg = np.average(np.abs(L - avg))
#   std_mde = np.average(np.abs(L - mde))
#   print("std:", std_avg, std_mde, np.min(L), mde, np.max(L))
#   return avg, std_avg, mde, std_mde, np.std(L)

name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)
posT = Xpos[1:] - Xpos[:-1]
negT = Xneg[1:] - Xneg[:-1]
T = np.hstack([posT, negT])

Xpos = hp.refine_frontier_iter(Xpos, W)
Xneg = hp.refine_frontier_iter(Xneg, W)
posT_ref = Xpos[1:] - Xpos[:-1]
negT_ref = Xneg[1:] - Xneg[:-1]
T_ref = np.hstack([posT_ref, negT_ref])

mode_error = T_ref - mode(T_ref)
modeX, modeY = np.unique(mode_error, return_counts=True)

avg_error = T_ref - np.average(T_ref)
avgX, avgY = np.unique(avg_error, return_counts=True)

'''============================================================================'''
'''                              PLOT LINES                                    '''
'''============================================================================'''

fig = fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="T",
  yaxis_title="Number of ocurrences",

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
    name=f"Mode (average absolute error={np.round(np.average(np.abs(mode_error)), 2)})", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=modeX,
    y=modeY,
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
    name=f"Average (average absolute error={np.round(np.average(np.abs(avg_error)), 2)})", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=avgX,
    y=avgY,
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