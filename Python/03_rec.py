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

def find_zeroes(V):
  s = np.sign(V[0])
  zeroes = []
  for i, v in enumerate(V):
    if np.sign(v) != s or v == 0:
      s = np.sign(-1 * s)
      zeroes.append(i)
  return np.array(zeroes, dtype=np.int)

'''==============='''
''' Read wav file '''
'''==============='''

name = "alto"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
print(f"n={n}")
X = np.arange(n)

Xpos_orig, Xneg_orig = se.get_frontiers(W)
Xpos = hp.refine_frontier_iter(Xpos_orig, W)
Xneg = hp.refine_frontier_iter(Xneg_orig, W)

'''find estimated Xpcs:'''
nn = min(Xpos.size, Xneg.size)
Average_ref_X = np.round((Xpos[:nn] + Xneg[:nn]) / 2)#.astype(np.int)

# Average_ref_X = np.round(savgol_filter(Average_ref_X, 51, 3)).astype(np.int) # <|<|<|<|

Average_ref_X_linear = hp.linearize_pc_approx(Average_ref_X)

nn = min(Average_ref_X_linear.size, Average_ref_X.size)
Xdev = Average_ref_X[:nn] - Average_ref_X_linear[:nn]
av = np.average(Xdev)
Xdev = Xdev - av

zeroes = find_zeroes(Xdev)


A = hp.constrained_least_squares_arbitrary_intervals(np.arange(nn), Xdev, zeroes.tolist(), 4)
Xdev_est = hp.coefs_to_array_arbitrary_intervals(A, X, zeroes.tolist(), Average_ref_X_linear.size)

Xpcs = np.round(np.sort(np.abs(Average_ref_X_linear + Xdev_est + av))).astype(np.int)



'''find average pc waveform:'''
avgpc, pos_orig_pcs, pos_norm_pcs = average_pc_waveform(np.round(np.abs(Average_ref_X)).astype(np.int), W) # TODO

pcx = interpolate.interp1d(np.linspace(0, 1, avgpc.size), avgpc, "cubic")
Wp = np.zeros(n)
for i in range(Xpcs.size - 1):
  x0 = Xpcs[i]
  x1 = Xpcs[i + 1]
  Wp[x0 : x1] = pcx(np.linspace(0, 1, x1 - x0))

# envelope 1
Wp = Wp / np.max(np.abs(Wp))
# dWp = np.zeros(Wp.size)
# dWp[: -1] = Wp[1 :] - Wp[: -1]
# intervals = hp.split(W, 10)
# # intervals[-1]=Wp.size - 1
# A = hp.constrained_least_squares_arbitrary_intervals_wtow(Wp, dWp, W[:Wp.size], intervals, 2)
# E = hp.coefs_to_array_arbitrary_intervals(A, np.arange(Wp.size), intervals, Wp.size)

# envelope 2
xf=np.unique(np.hstack([Xpos, Xneg]))
intervals = hp.split_raw_frontier(xf, W, n_intervals = 12)
A = hp.constrained_least_squares_arbitrary_intervals(xf, np.abs(W), intervals, 2)
E = hp.coefs_to_array_arbitrary_intervals(A, X, xf[intervals].tolist(), n)

se.save_wav(E * Wp, "Wp.wav", fps=fps)

'''============================================================================'''
'''                              PLOT Xpcs                                     '''
'''============================================================================'''
if True:
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
      y=Xpcs,
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
'''                              PLOT Signal                                   '''
'''============================================================================'''
if True:
  fig = fig = go.Figure()
  fig.layout.template ="plotly_white"
  # fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
  fig.update_layout(
    # title = name,
    xaxis_title="Length",
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

  fig.add_trace(
    go.Scatter(
      name="E", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=Xpos,
      y=E,
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
      name="Raw F", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=xf,
      y=np.abs(W[xf]),
      # fill="toself",
      mode="lines",
      line=dict(
          width=1,
          color="blue",
          # showscale=False
      ),
      visible = "legendonly"
    )
  )

  fig.add_trace(
    go.Scatter(
      name="Wp * E", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=xf,
      y=Wp * E,
      # fill="toself",
      mode="lines",
      line=dict(
          width=1,
          color="green",
          # showscale=False
      ),
      visible = "legendonly"
    )
  )

  # fig.add_trace(
  #   go.Scatter(
  #     name="dWp", # <|<|<|<|<|<|<|<|<|<|<|<|
  #     # x=Xpos,
  #     y=dWp,
  #     # fill="toself",
  #     mode="lines",
  #     line=dict(
  #         width=1,
  #         color="red",
  #         # showscale=False
  #     ),
  #     visible = "legendonly"
  #   )
  # )

  X = []
  Y = []
  for x in xf[intervals]:
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
          color="black",
          # showscale=False
      ),
      visible = "legendonly"
    )
  )

  fig.show(config=dict({'scrollZoom': True}))


'''============================================================================'''
'''                              PLOT LINES                                    '''
'''============================================================================'''
if False:
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
      y=avgpc,
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
      x=np.arange(avgpc.size) + avgpc.size - 1,
      y=avgpc,
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