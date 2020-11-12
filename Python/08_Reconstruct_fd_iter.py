import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp

def loop(W):
  n = W.size
  X = np.arange(n)
  '''get and refine frontiers '''
  Xpos, Xneg = se.get_frontiers(W)
  Xpos = hp.refine_frontier_iter(Xpos, W)
  Xneg = hp.refine_frontier_iter(Xneg, W)

  pcaverage = hp.average_pc_waveform(Xpos, Xneg, W)
  Ap = hp.approximate_pseudocycles_average(pcaverage)
  Wp, dWp = hp.parametric_W_wtow(Xpos, Ap, n)

  Xf = np.unique(np.hstack([Xpos, Xneg]))
  Ix = hp.split_raw_frontier(Xf, W, 10)

  A = hp.constrained_least_squares_arbitrary_intervals_wtow(Wp, dWp, W, Xf[Ix].tolist(), 2)
  We = hp.coefs_to_array_arbitrary_intervals_wtow(A, Wp, Xf[Ix].tolist(), n)
  residue = W - We

  env = hp.coefs_to_array_arbitrary_intervals(A, X, Xf[Ix].tolist(), n)
  Ws = hp.parametric_W(hp.linearize_pc_approx(Xpos), Ap, n, True)

  return We, residue


'''==============='''
''' Read wav file '''
'''==============='''

name = "alto"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)

compositeW = 0
Residues = []
for i in range(3):
  We, W = loop(W)
  compositeW += We
  se.save_wav(We, f"+0{i+1}_{name}.wav", fps=fps)
  # W = residue

se.save_wav(compositeW, f"+Final_{name}.wav", fps=fps)

exit()
'''============================================================================'''
'''                                    PLOT                                    '''
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
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
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
    name="Reconstructed W step 1", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=We_1,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Reconstructed W step 2", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=We_2,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="blue",
        # dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)


fig.show(config=dict({'scrollZoom': True}))
exit()
'''============================================================================'''
'''                                     FT                                     '''
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
  go.Scatter(
    name="W", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=np.abs(np.fft.rfft(W)),
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
    name="Reconstructed W", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=np.abs(np.fft.rfft(We)),
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="error W", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=X,
#     y=np.abs(np.fft.rfft(error)),
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="blue",
#         # dash="dash"
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

fig.show(config=dict({'scrollZoom': True}))
