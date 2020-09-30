import plotly.graph_objects as go
# import random
import numpy as np
from Helper import read_wav
# from scipy.signal import savgol_filter
import numpy.polynomial.polynomial as poly

class Pulse:
  def __init__(self, x0, W, isnoise = False):
    self.start = x0
    self.len = W.size
    self.end = self.start + self.len
    idx = np.argmax(np.abs(W)) #
    self.y = W[idx]            # np.average(W)
    self.x = idx               # np.sum(np.linspace(0, 1, self.len) * np.abs(W) / np.sum(np.abs(W)))
    self.noise = isnoise

'''Read wav file'''
name = "tom"
W, fps = read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude

# W = W [ : W.size // 2]

n = W.shape[0]
X = np.arange(n)

'''Split the signal into pulses'''
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

'''Mark pulses as noise based on length'''
for p in pulses:
  if p.len <=10:
    p.noise = True

# pos_pulses = [p for p in pulses if (np.sign(p.y) >= 0 and not p.noise)]
# neg_pulses = [p for p in pulses if (np.sign(p.y) < 0 and not p.noise)]


## Plot original signal, pulses and amplitude
X_pulses_area = []
Y_pulses_area = []

X_pos_pulses = []
Y_pos_pulses = []
X_pos_noise = []
Y_pos_noise = []

X_neg_pulses = []
Y_neg_pulses = []
X_neg_noise = []
Y_neg_noise = []

for p in pulses:
  X_pulses_area.append(p.start - .5)
  X_pulses_area.append(p.end - .5)
  X_pulses_area.append(None)
  Y_pulses_area.append(p.y)
  Y_pulses_area.append(p.y)
  Y_pulses_area.append(None)
  if p.noise:
    if p.y >= 0:
      X_pos_noise.append(p.start + p.x)
      Y_pos_noise.append(p.y)
    else:
      X_neg_noise.append(p.start + p.x)
      Y_neg_noise.append(p.y)
  else:
    if p.y >= 0:
      X_pos_pulses.append(p.start + p.x)
      Y_pos_pulses.append(p.y)
    else:
      X_neg_pulses.append(p.start + p.x)
      Y_neg_pulses.append(p.y)

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

fig.add_trace(
  go.Scatter(
    name="Pulses",
    x=X_pulses_area,
    y=Y_pulses_area,
    fill="tozeroy",
    mode="none",
    fillcolor="rgba(0,0,0,0.16)"
  )
)

fig.add_trace(
  go.Scatter(
    x=X_pos_pulses,
    y=Y_pos_pulses,
    # hovertext=np.arange(len(pos_pulses)),
    name="Positive Average Amplitude",
    mode="markers",
    marker=dict(
        size=3,
        color="black",
        # showscale=False
    )
  )
)

# X_neg_pulses = np.array([p.start + p.x for p in neg_pulses])
# Y_neg_pulses = np.array([p.y for p in neg_pulses])

fig.add_trace(
  go.Scatter(
    x=X_neg_pulses,
    y=Y_neg_pulses,
    # hovertext=np.arange(len(pos_pulses)),
    name="Negative Average Amplitude",
    mode="markers",
    marker=dict(
        size=3,
        color="black",
        # showscale=False
    )
  )
)

# plot positive and negative averages
terms = 4
pos_args = poly.polyfit(X_pos_pulses, Y_pos_pulses, terms)
neg_args = poly.polyfit(X_neg_pulses, Y_neg_pulses, terms)

X_pulses = np.array([p.start + p.x for p in pulses])
fitted_Y_pos_pulses = poly.polyval(X_pos_pulses, pos_args)
fitted_Y_neg_pulses = poly.polyval(X_neg_pulses, neg_args)

fig.add_trace(
  go.Scatter(
    x=X_pos_pulses,
    y=fitted_Y_pos_pulses,
    # hovertext=np.arange(ratios.size),
    name="Positive Average",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        # dash="dash" # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    x=X_neg_pulses,
    y=fitted_Y_neg_pulses,
    # hovertext=np.arange(ratios.size),
    name="Negative Average",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        # dash="dash"
        # showscale=False
    )
  )
)

pos_std = []
for a, b in zip(Y_pos_pulses, fitted_Y_pos_pulses):
  aa = abs(a)
  bb = abs(b)
  pos_std.append(min(aa, bb) / max(aa, bb))

pos_std = np.average(pos_std)

fig.add_trace(
  go.Scatter(
    x=X_pos_pulses,
    y=fitted_Y_pos_pulses - fitted_Y_pos_pulses * pos_std,
    # hovertext=np.arange(ratios.size),
    name="Positive std",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        dash="dash" # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        # showscale=False
    )
  )
)

neg_std = []
for a, b in zip(Y_neg_pulses, fitted_Y_neg_pulses):
  aa = abs(a)
  bb = abs(b)
  neg_std.append(min(aa, bb) / max(aa, bb))

neg_std = np.average(neg_std)


fig.add_trace(
  go.Scatter(
    x=X_neg_pulses,
    y=fitted_Y_neg_pulses - fitted_Y_neg_pulses * neg_std,
    # hovertext=np.arange(ratios.size),
    name="Negative std",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        dash="dash" # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        # showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)
