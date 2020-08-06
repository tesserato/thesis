import plotly.graph_objects as go
import numpy as np
from Helper import read_wav, signal_to_pulses, get_pulses_area, split_pulses, get_frontier#, save_wav

'''==============='''
''' Read wav file '''
'''==============='''
fig = go.Figure()

name = "alto"
W, fps = read_wav(f"Samples/{name}.wav")

# W = W [70000 : 100000]

W = W - np.average(W)
amplitude = np.max(np.abs(W))
# W = W / amplitude
n = W.size
print(n)
X = np.arange(n)



pulses = signal_to_pulses(W)

areas_X, areas_Y = get_pulses_area(pulses)

pos_pulses, neg_pulses, pos_noises, neg_noises = split_pulses(pulses)

pos_X, neg_X = np.array([p.x for p in pos_pulses]), np.array([p.x for p in neg_pulses])
pos_Y, neg_Y = np.array([p.y for p in pos_pulses]), np.array([p.y for p in neg_pulses])
pos_L, neg_L = np.array([p.len for p in pos_pulses]) , np.array([p.len for p in neg_pulses])

'''pos frontier'''
scaling = np.average(pos_L) / np.average(pos_Y)
# W = W * scaling
pos_Y = pos_Y * scaling
pos_frontier_X, pos_frontier_Y = get_frontier(pos_X, pos_Y, n)
pos_frontier_Y = pos_frontier_Y / scaling

'''neg frontier'''
scaling = np.average(neg_L) / np.average(neg_Y)
# W = W * scaling
neg_Y = neg_Y * scaling
neg_frontier_X, neg_frontier_Y = get_frontier(neg_X, neg_Y, n)
neg_frontier_Y = -neg_frontier_Y / scaling


frontier = (pos_frontier_Y + neg_frontier_Y) / 2

# # f = 500
# # recreated = np.cos(2 * np.pi * f * X / n) * frontier * 100

# freqs = irfft(np.random.normal(0, 1000, n // 2)) 

# recreated = freqs * frontier[:freqs.size]

# save_wav(recreated, f"Samples/{name}_r.wav")

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''


fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",

  # yaxis = dict(
  #   scaleanchor = "x",
  #   scaleratio = 1,
  # ),

  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

''' Signal '''
fig.add_trace(
  go.Scatter(
    name="Signal",
    x=X,
    y=W,
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

# analytic_signal = hilbert(W)
# amplitude_envelope = np.abs(analytic_signal)

# fig.add_trace(
#   go.Scatter(
#     name="Hilbert",
#     x=X,
#     y=amplitude_envelope,
#     mode="lines",
#     line=dict(
#         # size=8,
#         color="orange",
#         # showscale=False
#     )
#   )
# )

''' Pulses '''
fig.add_trace(
  go.Scatter(
    name="Pulses",
    x=areas_X,
    y=areas_Y,
    fill="tozeroy",
    mode="none",
    fillcolor="rgba(0,0,0,0.16)",
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+ Amplitudes",
    x=pos_X,
    y=pos_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="markers",
    marker=dict(
        size=6,
        color="black",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="- Amplitudes",
    x=neg_X,
    y=neg_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="markers",
    marker=dict(
        size=6,
        color="black",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Frontier",
    x=X,
    y=frontier,
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
    name="+ Frontier",
    x=pos_frontier_X,
    y=pos_frontier_Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="- Frontier",
    x=neg_frontier_X,
    y=-neg_frontier_Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Knots",
#     x=np.array([[pos_frontier_X[i], pos_frontier_X[i], None] for i in II]).flat,
#     y=np.array([[-1, 1, None] for _ in II]).flat,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="Pos Envelope Avgs",
#     x=XX,
#     y=YY,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         # width=1,
#         color="blue",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

fig.show(config=dict({'scrollZoom': True}))



'''============================================================================'''
'''                                     FT                                     '''
'''============================================================================'''

FT = np.abs(np.fft.rfft(W))
FT = FT / np.max(FT)
FT_normalized = np.abs(np.fft.rfft(W / frontier))
FT_normalized = FT_normalized / np.max(FT_normalized)

fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",
  # yaxis = dict(
  #   scaleanchor = "x",
  #   scaleratio = 1,
  # ),
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
    name="Original FT",
    # x=pos_X,
    y=FT,
    # hovertext=np.arange(len(pos_pulses)),
    mode="lines",
    line=dict(
        # size=6,
        color="black",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="Normalized FT",
    # x=pos_X,
    y=FT_normalized,
    # hovertext=np.arange(len(pos_pulses)),
    mode="lines",
    line=dict(
        # size=6,
        color="red",
        # showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)
