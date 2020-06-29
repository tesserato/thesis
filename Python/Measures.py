import plotly.graph_objects as go
import random
import numpy as np
from Helper import read_wav
from scipy.signal import savgol_filter

class Pulse:
  # pulses = []
  # max_length = 0
  def __init__(self, x0, W):
    # Pulse.pulses.append(self)
    self.start = x0 - .5
    self.end = self.start + W.shape[0]
    self.n = self.end - self.start
    self.W = W
    self.normalized_W = W / np.max(np.abs(W))
    self.avg = np.average(W)
    centered_W = W - np.average(W)
    FT = np.fft.rfft(centered_W)
    self.f = np.argmax(np.abs(FT))
    self.p = np.angle(FT[self.f])
    sign = np.sign(W[0])
    self.zeros = 0
    for w in centered_W:
      if sign != np.sign(w):
        self.zeros += 1
        sign = np.sign(w)


W, fps = read_wav("Samples/piano33.wav")
W = W - np.average(W)
a = np.max(np.abs(W))
W = W / a

# W = savgol_filter(W, 5, 3)

W = W [ : 5000]

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

## Plot original signal and pulses
XX = []
YY = []
for p in pulses:
  XX.append(p.start)
  XX.append(p.end)
  XX.append(None)
  YY.append(p.avg)
  YY.append(p.avg)
  YY.append(None)

fig = go.Figure()

fig.update_layout(
    # width = 2 * np.pi * 220,
    # height = n * 220,
    # yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title=f"Cycles",
    xaxis_title="x",
    yaxis_title="Amplitude",
    # font=dict(
    #     family="Courier New, monospace",
    #     size=18,
    #     color="#7f7f7f"
    # )
)

fig.add_trace(
  go.Scatter(
    x=X,
    y=W,
    name="Signal W[x]",
    mode="lines",
    # marker=dict(
    #     size=8,
    #     color="red", #set color equal to a variable
    #     showscale=False
    # )
  )
)

fig.add_trace(
  go.Scatter(
    x=XX,
    y=YY,
    fill="tozeroy",
    name=f"Half Cycles",
    mode="none",
    # marker=dict(
    #     size=8,
    #     color="black", #set color equal to a variable
    #     showscale=False
    # )
  )
)

fig.show(config=dict({'scrollZoom': True}))

## Plot f x p
XX = []
YY = []
for p in pulses:
  XX.append(p.p)
  YY.append(p.f)

fig = go.Figure()

fig.update_layout(
    # width = 2 * np.pi * 220,
    # height = n * 220,
    # yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title=f"Frequency x Phase",
    xaxis_title="Phase",
    yaxis_title="Frequency",
    # font=dict(
    #     family="Courier New, monospace",
    #     size=18,
    #     color="#7f7f7f"
    # )
)

fig.add_trace(
  go.Scatter(
    x=XX,
    y=YY,
    name="",
    mode="markers",
    # marker=dict(
    #     size=8,
    #     color="black", #set color equal to a variable
    #     showscale=False
    # )
  )
)

fig.show(config=dict({'scrollZoom': True}))

