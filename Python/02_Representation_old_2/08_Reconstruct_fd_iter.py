import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp
from scipy import interpolate


def average_pc_waveform(Xpos, Xneg, W):
  T = []
  for i in range(1, Xpos.size):
    T.append(Xpos[i] - Xpos[i - 1])
  for i in range(1, Xneg.size):
    T.append(Xneg[i] - Xneg[i - 1])
  T = np.array(T, dtype = np.int)
  maxT = np.max(T)

  Xlocal = np.linspace(0, 1, maxT)

  pos_orig_pcs = []
  pos_norm_pcs = []
  for i in range(1, Xpos.size):
    x0 = Xpos[i - 1]
    x1 = Xpos[i]
    pos_orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      pos_norm_pcs.append(Ylocal)

  neg_orig_pcs = []
  neg_norm_pcs = []
  for i in range(1, Xneg.size):
    x0 = Xneg[i - 1]
    x1 = Xneg[i]
    neg_orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      neg_norm_pcs.append(Ylocal)
  pos_avgpc = np.average(np.array(pos_norm_pcs), 0)
  return pos_avgpc


def loop(W, Xpos, Xneg):
  n = W.size
  X = np.arange(n)

  #############
  Xpos, Xneg = se.get_frontiers(W)
  Xpos = hp.refine_frontier_iter(Xpos, W)
  Xneg = hp.refine_frontier_iter(Xneg, W)
  #############

  pcaverage = average_pc_waveform(Xpos, Xneg, W)

  # Ap = hp.approximate_pseudocycles_average(pcaverage)
  # Wp, dWp = hp.parametric_W_wtow(Xpos, Ap, n)

  pcx = interpolate.interp1d(np.linspace(0, 1, pcaverage.size), pcaverage, "cubic")
  Wp = np.zeros(n)
  for i in range(Xpos.size - 1):
    x0 = Xpos[i]
    x1 = Xpos[i + 1]
    Wp[x0 : x1] = pcx(np.linspace(0, 1, x1 - x0))
  dWp = np.zeros(n)
  dWp[:-1] = Wp[1 :] - Wp[:-1]

  Xf = np.unique(np.hstack([Xpos, Xneg]))
  Ix = hp.split_raw_frontier(Xf, W, 10)

  A = hp.constrained_least_squares_arbitrary_intervals_wtow(Wp, dWp, W, Xf[Ix].tolist(), 2)
  We = hp.coefs_to_array_arbitrary_intervals_wtow(A, Wp, Xf[Ix].tolist(), n)
  residue = W - We

  env = hp.coefs_to_array_arbitrary_intervals(A, X, Xf[Ix].tolist(), n)
  # Ws = hp.parametric_W(hp.linearize_pc(Xpos), Ap, n, True)

  Xposl = hp.linearize_pc(Xpos)
  Ws = np.zeros(n)
  for i in range(1, Xposl.size - 1):
    # print(i)
    x0 = Xposl[i]
    x1 = Xposl[i + 1]
    Ws[x0 : x1] = pcx(np.linspace(0, 1, x1 - x0))

  return We, Ws * env, residue


'''==============='''
''' Read wav file '''
'''==============='''

name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)

'''get and refine frontiers '''
Xpos, Xneg = se.get_frontiers(W)
Xpos = hp.refine_frontier_iter(Xpos, W)
Xneg = hp.refine_frontier_iter(Xneg, W)

compositeW = 0
WWe = []
WWl = []
residues = [W]
for i in range(3):
  We, Wl, W = loop(W, Xpos, Xneg)
  compositeW += Wl
  se.save_wav(Wl, f"+0{i+1}_{name}.wav", fps=fps)
  WWe.append(We)
  WWl.append(Wl)
  residues.append(W)

se.save_wav(residues[1], "res.wav", fps=fps)

se.save_wav(compositeW, f"+Final_{name}.wav", fps=fps)


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
    y=residues[0],
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
    name="Reconstructed", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=compositeW,
    mode="lines",
    line=dict(
        # size=8,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

for i, w in enumerate(WWl):
  fig.add_trace(
    go.Scatter(
      name=f"Rec W linear {i + 1}", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=X,
      y=w,
      # fill="toself",
      mode="lines",
      line=dict(
          width=1,
          color="red",
          # dash="dash"
          # showscale=False
      ),
      visible = "legendonly"
    )
  )

for i, w in enumerate(WWe):
  fig.add_trace(
    go.Scatter(
      name=f"Rec W {i + 1}", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=X,
      y=w,
      # fill="toself",
      mode="lines",
      line=dict(
          width=1,
          color="black",
          # dash="dash"
          # showscale=False
      ),
      # visible = "legendonly"
    )
  )

for i in range(1, len(residues)):
  fig.add_trace(
    go.Scatter(
      name=f"Residues {i}", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=X,
      y=residues[i],
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
