import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import numpy.polynomial.polynomial as poly
# import numpy.polynomial.polynomial as poly
# from collections import Counter
# from math import gcd
import hp as hp
from scipy.signal import savgol_filter
# from plotly.subplots import make_subplots
from scipy import interpolate

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

def average_pc_waveform(Xp, W):
  maxT = np.max(np.abs(Xp[1:] - Xp[:-1]))

  Xlocal = np.linspace(0, 1, maxT)

  pos_orig_pcs = []
  pos_norm_pcs = []
  for i in range(1, Xp.size):
    x0 = Xp[i - 1]
    x1 = Xp[i] + 1
    pos_orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
      # Ylocal = yx(Xlocal)
      pos_norm_pcs.append(yx(Xlocal))

  pos_avgpc = np.average(np.array(pos_norm_pcs), 0)
  return pos_avgpc, pos_orig_pcs, pos_norm_pcs


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
Xneg = hp.refine_frontier_iter(Xneg_orig, W)

siz = min(Xpos.size, Xneg.size)
Average_ref_X = (Xpos[:siz] + Xneg[:siz]) / 2
Average_ref_X_smooth = np.round(savgol_filter(Average_ref_X, 51, 3)).astype(np.int)
Average_ref_X_smooth = np.unique(np.abs(Average_ref_X_smooth))

Average_ref_X_smooth_linear = hp.linearize_pc_approx(Average_ref_X_smooth)

pos_avgpc, pos_orig_pcs, pos_norm_pcs = average_pc_waveform(Average_ref_X_smooth, W)

pcx = interpolate.interp1d(np.linspace(0, 1, pos_avgpc.size), pos_avgpc, "cubic")
Wp = np.zeros(n)
for i in range(Average_ref_X_smooth_linear.size - 1):
  x0 = Average_ref_X_smooth_linear[i]
  x1 = Average_ref_X_smooth_linear[i + 1]
  Wp[x0 : x1] = pcx(np.linspace(0, 1, x1 - x0))


se.save_wav(Wp, "Wp.wav", fps=fps)

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
    visible = "legendonly"
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
    visible = "legendonly"
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

fig.add_trace(
  go.Scatter(
    name="Average Refined Extrema", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Average_ref_X,
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
    name="Average Refined Extrema Smooth", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Average_ref_X_smooth,
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

fig.add_trace(
  go.Scatter(
    name="Average Refined Extrema Smooth Linear", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Average_ref_X_smooth_linear,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="green",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)
fig.show(config=dict({'scrollZoom': True}))


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
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=W,
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

X = []
Y = []
for x in Average_ref_X:
  X.append(x)
  X.append(x)
  X.append(None)
  Y.append(-amp)
  Y.append(amp)
  Y.append(None)

fig.add_trace(
  go.Scatter(
    name="divs", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
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

X = []
Y = []
for x in Average_ref_X_smooth:
  X.append(x)
  X.append(x)
  X.append(None)
  Y.append(-amp)
  Y.append(amp)
  Y.append(None)

fig.add_trace(
  go.Scatter(
    name="divs smooth", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
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

transp_black = "rgba(38, 12, 12, 0.2)"

X, Y = to_plot(pos_orig_pcs)
fig.add_trace(
  go.Scattergl(
    name="pos_orig_pcs", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color=transp_black,
        # showscale=False
    ),
    visible = "legendonly"
  )
)

X, Y = to_plot(pos_norm_pcs)
fig.add_trace(
  go.Scattergl(
    name="pos_norm_pcs", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color=transp_black,
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="Average", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=pos_avgpc,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="Average shift", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=np.arange(pos_avgpc.size) + pos_avgpc.size - 1,
    y=pos_avgpc,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))