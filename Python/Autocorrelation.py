import plotly.graph_objects as go
import random
import numpy as np
from Helper import read_wav
from scipy.signal import savgol_filter

class Pulse:
  def __init__(self, x0, W):
    self.start = x0
    self.n = W.size
    self.end = self.start + self.n
    self.my = np.average(W / np.max(np.abs(W)))
    self.mx = np.sum(np.linspace(0, 1, self.n) * np.abs(W) / np.sum(np.abs(W)))
    # print(self.n, self.mx)


W, fps = read_wav("Samples/piano33.wav")
W = W - np.average(W)
a = np.max(np.abs(W))
W = W / a

# W = W [ : W.size // 4]

n = W.shape[0]
X = np.arange(n)

## Split the signal into pulses
pulses = []
sign = np.sign(W[0])
x0 = 0
max_length = 0
for x in range(n):
  if sign != np.sign(W[x]):
    # w = W[x0 : x]
    pulses.append(Pulse(x0, W[x0 : x]))
    if x - x0 > max_length:
      max_length = x - x0
    x0 = x
    sign = np.sign(W[x])

pulses.append(Pulse(x0, W[x0 : n]))
if n - x0 > max_length:
  max_length = n - x0

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
  X_avg.append((p.start + p.mx * p.n) - .5)
  Y_avg.append(p.my)

fig = go.Figure()
fig.layout.template ="plotly_white"
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
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

fig.add_trace(
  go.Scatter(
    x=X_avg,
    y=Y_avg,
    name="Pulses",
    mode="markers",
    marker=dict(
        # size=8,
        color="black",
        # showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)

# abs_FT = np.abs(np.fft.rfft(Y_avg))[:-1]
# # correlation = np.correlate(Y_avg[ : nY_avg//4], Y_avg, "valid")

# fig = go.Figure()
# fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
# fig.update_layout(
#   xaxis_title="Frequency",
#   yaxis_title="Intensity",
#   legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
#   margin=dict(l=5, r=5, b=5, t=5),
#   font=dict(
#   family="Computer Modern",
#   color="black",
#   size=18
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     # x=X,
#     y=abs_FT,
#     name="Signal",
#     mode="lines",
#     line=dict(
#         # size=8,
#         color="black",
#         # showscale=False
#     )
#   )
# )

# fig.show(config=dict({'scrollZoom': True}))
