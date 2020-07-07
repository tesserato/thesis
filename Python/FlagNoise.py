import plotly.graph_objects as go
# import random
import numpy as np
from Helper import read_wav
# from scipy.signal import savgol_filter
import numpy.polynomial.polynomial as poly

class Pulse:
  def __init__(self, x0, W, is_noise = False):
    self.start = x0
    self.len = W.size
    self.end = self.start + self.len
    idx = np.argmax(np.abs(W)) #
    self.y = W[idx]            # np.average(W)
    self.x = idx               # np.sum(np.linspace(0, 1, self.len) * np.abs(W) / np.sum(np.abs(W)))
    self.noise = is_noise

'''==============='''
''' Read wav file '''
'''==============='''

name = "piano33"
W, fps = read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude

# W = W [ : W.size // 2]

n = W.shape[0]
X = np.arange(n)


'''=============================='''
''' Split the signal into pulses '''
'''=============================='''

pulses = []
sign = np.sign(W[0])
x0 = 0
max_length = 0
for x in range(n):
  if sign != np.sign(W[x]):
    p = Pulse(x0, W[x0 : x])
    pulses.append(p)
    if x - x0 > max_length:
      max_length = x - x0
    x0 = x
    assert(np.all(np.sign(W[x0 : x]) == sign))
    sign = np.sign(W[x])

pulses.append(Pulse(x0, W[x0 : n]))
if n - x0 > max_length:
  max_length = n - x0

print(f"Max length of a pulse={max_length}")


'''============================================================================='''
''' Mark pulses as noise based on Nyquist and divide into positive and negative '''
'''============================================================================='''

pos_noises = []
pos_pulses = []
neg_noises = []
neg_pulses = []
for p in pulses:
  if p.len <=2:
    p.noise = True
    if p.y >= 0:
      pos_noises.append(p)
    else:
      neg_noises.append(p)
  else:
    if p.y >= 0:
      pos_pulses.append(p)
    else:
      neg_pulses.append(p)

print(f"{len(pos_pulses) + len(neg_pulses)} of {len(pulses)} are valid (lenght > 2)")


'''===================================='''
''' Make plotly figure and plot signal '''
'''===================================='''

fig = go.Figure()
fig.layout.template ="plotly_white"
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",
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
    name="Signal",
    x=X,
    y=W,
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    )
  )
)


'''============================================='''
''' Pulses area and length and amplitude vectors'''
'''============================================='''

pos_pulses_area_X = []
pos_pulses_area_Y = []
neg_pulses_area_X = []
neg_pulses_area_Y = []

pos_pulses_X = []
pos_pulses_Y = []
pos_pulses_len = []
neg_pulses_X = []
neg_pulses_Y = []
neg_pulses_len = []

for p in pos_pulses:
  pos_pulses_area_X.append(p.start - .5)
  pos_pulses_area_Y.append(0)
  pos_pulses_area_X.append(p.start - .5)
  pos_pulses_area_Y.append(p.y)
  pos_pulses_area_X.append(p.end - .5)
  pos_pulses_area_Y.append(p.y)
  pos_pulses_area_X.append(p.end - .5)
  pos_pulses_area_Y.append(0)
  pos_pulses_area_X.append(None)
  pos_pulses_area_Y.append(None)

  pos_pulses_X.append(p.start + p.x)
  pos_pulses_Y.append(p.y)
  pos_pulses_len.append(p.len)

for p in neg_pulses:
  neg_pulses_area_X.append(p.start - .5)
  neg_pulses_area_Y.append(0)
  neg_pulses_area_X.append(p.start - .5)
  neg_pulses_area_Y.append(p.y)
  neg_pulses_area_X.append(p.end - .5)
  neg_pulses_area_Y.append(p.y)
  neg_pulses_area_X.append(p.end - .5)
  neg_pulses_area_Y.append(0)
  neg_pulses_area_X.append(None)
  neg_pulses_area_Y.append(None)

  neg_pulses_X.append(p.start + p.x)
  neg_pulses_Y.append(p.y)
  neg_pulses_len.append(p.len)

fig.add_trace(
  go.Scatter(
    name="Positive Pulses",
    x=pos_pulses_area_X,
    y=pos_pulses_area_Y,
    fill="tozeroy",
    mode="none",
    fillcolor="rgba(0,0,0,0.16)"
  )
)
fig.add_trace(
  go.Scatter(
    name="Negative Pulses",
    x=neg_pulses_area_X,
    y=neg_pulses_area_Y,
    fill="tozeroy",
    mode="none",
    fillcolor="rgba(0,0,0,0.16)"
  )
)


fig.add_trace(
  go.Scatter(
    x=pos_pulses_X,
    y=pos_pulses_Y,
    # hovertext=np.arange(len(pos_pulses)),
    name="Positive Amplitudes",
    mode="markers",
    marker=dict(
        size=3,
        color="black",
        # showscale=False
    )
  )
)
fig.add_trace(
  go.Scatter(
    x=neg_pulses_X,
    y=neg_pulses_Y,
    # hovertext=np.arange(len(neg_pulses)),
    name="Negative Amplitudes",
    mode="markers",
    marker=dict(
        size=3,
        color="black",
        # showscale=False
    )
  )
)

pos_pulses_len = np.array(pos_pulses_len)
pos_pulses_len = pos_pulses_len / np.max(pos_pulses_len)
fig.add_trace(
  go.Scatter(
    name="Positive Lengths",
    x=pos_pulses_X,
    y=pos_pulses_len,
    # hovertext=np.arange(len(pos_pulses)),
    mode="markers",
    marker=dict(
        size=3,
        color="red",
        # showscale=False
    )
  )
)
neg_pulses_len = np.array(neg_pulses_len)
neg_pulses_len = neg_pulses_len / np.max(neg_pulses_len)
fig.add_trace(
  go.Scatter(
    name="Negative Lengths",
    x=neg_pulses_X,
    y=-neg_pulses_len,
    # hovertext=np.arange(len(neg_pulses)),
    mode="markers",
    marker=dict(
        size=3,
        color="red",
        # showscale=False
    )
  )
)

# X_neg_pulses = np.array([p.start + p.x for p in neg_pulses])
# Y_neg_pulses = np.array([p.y for p in neg_pulses])

# fig.add_trace(
#   go.Scatter(
#     x=X_neg_pulses,
#     y=Y_neg_pulses,
#     # hovertext=np.arange(len(pos_pulses)),
#     name="Negative Average Amplitude",
#     mode="markers",
#     marker=dict(
#         size=3,
#         color="black",
#         # showscale=False
#     )
#   )
# )

# # plot positive and negative averages
# terms = 4
# pos_args = poly.polyfit(X_pos_pulses, Y_pos_pulses, terms)
# neg_args = poly.polyfit(X_neg_pulses, Y_neg_pulses, terms)

# X_pulses = np.array([p.start + p.x for p in pulses])
# fitted_Y_pos_pulses = poly.polyval(X_pos_pulses, pos_args)
# fitted_Y_neg_pulses = poly.polyval(X_neg_pulses, neg_args)

# fig.add_trace(
#   go.Scatter(
#     x=X_pos_pulses,
#     y=fitted_Y_pos_pulses,
#     # hovertext=np.arange(ratios.size),
#     name="Positive Average",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # dash="dash" # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
#         # showscale=False
#     )
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     x=X_neg_pulses,
#     y=fitted_Y_neg_pulses,
#     # hovertext=np.arange(ratios.size),
#     name="Negative Average",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # dash="dash"
#         # showscale=False
#     )
#   )
# )

# pos_std = []
# for a, b in zip(Y_pos_pulses, fitted_Y_pos_pulses):
#   aa = abs(a)
#   bb = abs(b)
#   pos_std.append(min(aa, bb) / max(aa, bb))

# pos_std = np.average(pos_std)

# fig.add_trace(
#   go.Scatter(
#     x=X_pos_pulses,
#     y=fitted_Y_pos_pulses - fitted_Y_pos_pulses * pos_std,
#     # hovertext=np.arange(ratios.size),
#     name="Positive std",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         dash="dash" # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
#         # showscale=False
#     )
#   )
# )

# neg_std = []
# for a, b in zip(Y_neg_pulses, fitted_Y_neg_pulses):
#   aa = abs(a)
#   bb = abs(b)
#   neg_std.append(min(aa, bb) / max(aa, bb))

# neg_std = np.average(neg_std)


# fig.add_trace(
#   go.Scatter(
#     x=X_neg_pulses,
#     y=fitted_Y_neg_pulses - fitted_Y_neg_pulses * neg_std,
#     # hovertext=np.arange(ratios.size),
#     name="Negative std",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         dash="dash" # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
#         # showscale=False
#     )
#   )
# )

fig.show(config=dict({'scrollZoom': True}))
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)
