import plotly.graph_objects as go
import random
import numpy as np
from Helper import read_wav
# from scipy.signal import savgol_filter
import numpy.polynomial.polynomial as poly

class Pulse:
  def __init__(self, x0, W):
    self.start = x0
    self.len = W.size
    self.end = self.start + self.len
    self.my = np.average(W)
    self.mx = np.sum(np.linspace(0, 1, self.len) * np.abs(W) / np.sum(np.abs(W)))
    # print(self.n, self.mx)

name = "alto"
W, fps = read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
a = np.max(np.abs(W))
W = W / a

W = W [ : 45000]

n = W.shape[0]
X = np.arange(n)

## Split the signal into pulses, positive pulses and negative pulses
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

pos_pulses = []
neg_pulses = []
for p in pulses:
  if np.sign(p.my) >= 0:
    pos_pulses.append(p)
  else:
    neg_pulses.append(p)

print(f"Max length of a pulse={max_length}")

## Plot original signal, pulses and amplitude
XX = []
YY = []
X_avg = []
Y_avg = []
for p in pulses:
  XX.append(p.start - .5)
  XX.append(p.end - .5)
  XX.append(None)
  YY.append(p.my)
  YY.append(p.my)
  YY.append(None)
  X_avg.append(p.start + p.mx * p.len)
  Y_avg.append(p.my)

fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
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
    x=X,
    y=W,
    name="Signal",
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
    x=XX,
    y=YY,
    fill="tozeroy",
    name="Pulses",
    mode="none",
    fillcolor="rgba(0,0,0,0.16)"
  )
)

X_pos_pulses = np.array([p.start + p.mx * p.len for p in pos_pulses])
Y_pos_pulses = np.array([p.my for p in pos_pulses])

fig.add_trace(
  go.Scatter(
    x=X_pos_pulses,
    y=Y_pos_pulses,
    hovertext=np.arange(len(pos_pulses)),
    name="+ Average Amplitude",
    mode="markers",
    marker=dict(
        # size=8,
        color="black",
        # showscale=False
    )
  )
)

X_neg_pulses = np.array([p.start + p.mx * p.len for p in neg_pulses])
Y_neg_pulses = np.array([p.my for p in neg_pulses])

fig.add_trace(
  go.Scatter(
    x=X_neg_pulses,
    y=Y_neg_pulses,
    hovertext=np.arange(len(pos_pulses)),
    name="- Average Amplitude",
    mode="markers",
    marker=dict(
        # size=8,
        color="black",
        # showscale=False
    )
  )
)

# plot positive and negative averages
terms = 3
pos_args = poly.polyfit(X_pos_pulses, Y_pos_pulses, terms)
neg_args = poly.polyfit(X_neg_pulses, Y_neg_pulses, terms)

X_pulses = np.array([p.start + p.mx * p.len for p in pulses])
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
        # size=8,
        color="blue",
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
        # size=8,
        color="blue",
        # dash="dash"
        # showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)
