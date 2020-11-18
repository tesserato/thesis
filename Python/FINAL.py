import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import helper as hp # <|<|<|
import numpy.polynomial.polynomial as poly
from scipy.signal import savgol_filter
from scipy import interpolate



'''==============='''
''' Read wav file '''
'''==============='''

name = "soprano"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
# W = W / amp
n = W.size
print(f"n={n}")
X = np.arange(n)

'''==========================='''
''' Find and refine Frontiers '''
'''==========================='''

Xpos_orig, Xneg_orig = se.get_frontiers(W)
Xpos = hp.refine_frontier_iter(Xpos_orig, W).astype(np.int)
Xneg = hp.refine_frontier_iter(Xneg_orig, W).astype(np.int)

n_Xpc = min(Xpos.size, Xneg.size)
Xpc = np.round((Xpos[:n_Xpc] + Xneg[:n_Xpc]) / 2).astype(np.int)
Xpc = Xpc[Xpc > 0]
# Xpc_smooth = np.round(savgol_filter(Xpc, 7, 3)).astype(np.int)
# Xpc_smooth = np.round(Xpc).astype(np.int)

# X_Xpc = np.linspace(0.0, 1.0, n_Xpc)#, dtype=np.float64)#
X_Xpc = np.arange(n_Xpc, dtype=np.float64)
b, a = poly.polyfit(X_Xpc, Xpc, 1)
Xpc_linear = a * X_Xpc + b

'''==========================='''
''' Find Xpc deviation '''
'''==========================='''
deviation_Xpc = Xpc - Xpc_linear
average_deviation_Xpc = np.average(deviation_Xpc)
deviation_Xpc = deviation_Xpc - average_deviation_Xpc

zeroes = hp.find_zeroes_alt(deviation_Xpc)

# zeroes = zeroes[:2] + [100, 123] + zeroes[2:]

A = hp.constrained_least_squares_arbitrary_intervals(X_Xpc, deviation_Xpc, zeroes, 4)
deviation_Xpc_est = hp.coefs_to_array_arbitrary_intervals(A, X_Xpc, zeroes, n_Xpc) #

Xpc_est = np.round(Xpc_linear + average_deviation_Xpc + deviation_Xpc_est).astype(np.int)
Xpc_est = Xpc_est[Xpc_est > 0]

'''==========================='''
''' Find average pc waveform: '''
'''==========================='''
# avgpc, orig_pcs, norm_pcs = hp.average_pc_waveform(Xpc.astype(np.int), W)
avgpc, orig_pcs, norm_pcs = hp.average_pc_waveform(Xpc, W)


I = hp.find_zeroes_alt(avgpc)
A = hp.approximate_pc_waveform(np.arange(avgpc.size), avgpc, I, 2, "k")
avgpc_est = hp.coefs_to_array_arbitrary_intervals(A, np.arange(avgpc.size), I, avgpc.size)

'''==========================='''
''' Reconstruct Wave: '''
'''==========================='''

pcx = interpolate.interp1d(np.linspace(0, 1, avgpc.size), avgpc, "cubic")
Wp = np.zeros(n)
for i in range(Xpc_est.size - 1):
  x0 = Xpc_est[i]
  x1 = Xpc_est[i + 1]
  Wp[x0 : x1] = pcx(np.linspace(0, 1, x1 - x0))
se.save_wav( Wp, "Wp.wav", fps=fps)


'''====================================================================================================================='''
transp_black = "rgba(38, 12, 12, 0.2)"
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

'''==========================='''
''' PLOT Signal & frontiers '''
'''==========================='''
if False:
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
    go.Scattergl(
      name="W", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=X,
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
    go.Scattergl(
      name="Max orig", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=Xpos_orig,
      y=W[Xpos_orig],
      # fill="toself",
      mode="markers",
      marker=dict(
          size=8,
          color="blue",
          # showscale=False
      ),
      # visible = "legendonly"
    )
  )

  fig.add_trace(
    go.Scattergl(
      name="Min orig", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=Xneg_orig,
      y=W[Xneg_orig],
      # fill="toself",
      mode="markers",
      marker=dict(
          size=8,
          color="blue",
          # showscale=False
      ),
      # visible = "legendonly"
    )
  )

  fig.add_trace(
    go.Scattergl(
      name="Max", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=Xpos,
      y=W[Xpos],
      # fill="toself",
      mode="markers",
      marker=dict(
          size=4,
          color="red",
          # showscale=False
      ),
      # visible = "legendonly"
    )
  )

  fig.add_trace(
    go.Scattergl(
      name="Min", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=Xneg,
      y=W[Xneg],
      # fill="toself",
      mode="markers",
      marker=dict(
          size=4,
          color="red",
          # showscale=False
      ),
      # visible = "legendonly"
    )
  )

  # XI = []
  # YI = []
  # maxI = np.max(np.abs(avgpc))
  # for i in I:
  #   XI. append(i), XI. append(i), XI. append(None)
  #   YI. append(-maxI), YI. append(maxI), YI. append(None)
  # fig.add_trace(
  #   go.Scattergl(
  #     name="Zeroes", # <|<|<|<|<|<|<|<|<|<|<|<|
  #     x=XI,
  #     y=YI,
  #     # fill="toself",
  #     mode="lines",
  #     line=dict(
  #         width=1,
  #         color="gray",
  #         # showscale=False
  #     ),
  #     # visible = "legendonly"
  #   )
  # )

  fig.show(config=dict({'scrollZoom': True}))


'''==========================='''
''' PLOT Average pcs waveform '''
'''==========================='''
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

  X, Y = to_plot(orig_pcs)
  fig.add_trace(
    go.Scattergl(
      name="Original Pseudo Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
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

  X, Y = to_plot(norm_pcs)
  fig.add_trace(
    go.Scattergl(
      name="Normalize Pseudo Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
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

  fig.add_trace(
    go.Scattergl(
      name="Average Waveform", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=np.arange(pos_avgpc.size) + pos_avgpc.size - 1,
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

  XI = []
  YI = []
  maxI = np.max(np.abs(avgpc))
  for i in I:
    XI. append(i), XI. append(i), XI. append(None)
    YI. append(-maxI), YI. append(maxI), YI. append(None)
  fig.add_trace(
    go.Scattergl(
      name="Zeroes", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=XI,
      y=YI,
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
    go.Scattergl(
      name="Average Waveform 2", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=np.arange(avgpc.size) + avgpc.size - 1,
      y=avgpc,
      # fill="toself",
      mode="lines",
      line=dict(
          width=2,
          color="red",
          dash="dot"
          # showscale=False
      ),
      visible = "legendonly"
    )
  )

  fig.add_trace(
    go.Scattergl(
      name="Estimated Waveform", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=np.arange(avgpc.size) + avgpc.size - 1,
      y=avgpc_est,
      # fill="toself",
      mode="lines",
      line=dict(
          width=2,
          color="blue",
          # dash="dot"
          # showscale=False
      ),
      visible = "legendonly"
    )
  )

  fig.add_trace(
    go.Scattergl(
      name="Estimated Waveform 2", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=np.arange(avgpc.size) + avgpc.size - 1,
      y=avgpc_est,
      # fill="toself",
      mode="lines",
      line=dict(
          width=2,
          color="blue",
          dash="dot"
          # showscale=False
      ),
      visible = "legendonly"
    )
  )

  fig.show(config=dict({'scrollZoom': True}))


'''=================================='''
''' PLOT Xpos, Xneg, Xpc and Xpc_est '''
'''=================================='''
if False:
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
    go.Scattergl(
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
      visible = "legendonly"
    )
  )

  fig.add_trace(
    go.Scattergl(
      name="Minima", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=Xpos,
      y=Xneg,
      # fill="toself",
      mode="lines",
      line=dict(
          width=1,
          color="gray",
          # showscale=False
      ),
      visible = "legendonly"
    )
  )

  fig.add_trace(
    go.Scattergl(
      name="Xpc", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=Xpos,
      y=Xpc,
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
    go.Scattergl(
      name="Xpc smooth", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=Xpos,
      y=Xpc,
      # fill="toself",
      mode="lines",
      line=dict(
          width=1,
          color="red",
          dash="dot"
          # showscale=False
      ),
      # visible = "legendonly"
    )
  )

  fig.add_trace(
    go.Scattergl(
      name="Xpc estimated", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=Xpos,
      y=Xpc_est,
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
    go.Scattergl(
      name="Xpc_linear", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=Xpos,
      y=Xpc_linear,
      # fill="toself",
      mode="lines",
      line=dict(
          width=1,
          color="black",
          dash="dot"
          # showscale=False
      ),
      visible = "legendonly"
    )
  )
  
  fig.show(config=dict({'scrollZoom': True}))


'''====================='''
''' PLOT Xpcs Deviation '''
'''====================='''
if False:
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
    go.Scattergl(
      name="deviation_Xpc", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=Xpos,
      y=deviation_Xpc,
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
      name="deviation_Xpc_est", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=Xpos,
      y=deviation_Xpc_est,
      # fill="toself",
      mode="lines+markers",
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
  ma = np.max(deviation_Xpc)
  mi = np.min(deviation_Xpc)
  for x in zeroes:
    X.append(x)
    X.append(x)
    X.append(None)
    Y.append(mi)
    Y.append(ma)
    Y.append(None)

  fig.add_trace(
    go.Scattergl(
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
      # visible = "legendonly"
    )
  )

  fig.show(config=dict({'scrollZoom': True}))



