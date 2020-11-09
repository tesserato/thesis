import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
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
Xpos = hp.refine_frontier_iter(Xpos, W)
Xneg = hp.refine_frontier_iter(Xneg, W)

Xf = np.unique(np.hstack([Xpos, Xneg]))

Ix = hp.split_raw_frontier(Xf, W, 10)
A = hp.constrained_least_squares_arbitrary_intervals(Xf, np.abs(W), Ix, 2)
E = hp.coefs_to_array_arbitrary_intervals(A, Xf, Ix, n)

# W = W / E

pcaverage, _, _ = hp.average_pc_waveform(Xpos, Xneg, W)
A = hp.approximate_pseudocycles_average(pcaverage)
# if used_positive_frontier:
#   Wp = hp.parametric_W(hp.linearize_pc(Xpos.astype(np.int)), A, n, False)
# else:
#   Wp = hp.parametric_W(hp.linearize_pc(Xneg.astype(np.int)), A, n, False)

Wp = hp.parametric_W(hp.linearize_pc(Xpos.astype(np.int)), A, n, False)
# Wp = hp.parametric_W(Xpos, A, n, False)

We = Wp * E

# W = W * E

se.save_wav(We)



Xintervals = []
Yintervals = []
for x in Xf[Ix]:
  Xintervals.append(x)
  Xintervals.append(x)
  Xintervals.append(None)
  Yintervals.append(-amp)
  Yintervals.append(amp)
  Yintervals.append(None)


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

fig.add_trace(
  go.Scatter(
    name="Raw Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xf,
    y=np.abs(W[Xf]),
    # fill="toself",
    mode="lines+markers",
    line=dict(
      width=1,
      color="gray",
      # showscale=False
    ),
    marker=dict(
      size=2,
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Breaks", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xintervals,
    y=Yintervals,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="gray",
        # dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Reconstructed W", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=We,
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
    name="Error", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W - We,
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