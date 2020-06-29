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
    self.a = np.average(W)
    # centered_W = W - np.average(W)
    # self.FT = np.fft.rfft(W)

    # sign = np.sign(W[0])
    # self.zeros = 0
    # for w in centered_W:
    #   if sign != np.sign(w):
    #     self.zeros += 1
    #     sign = np.sign(w)


W, fps = read_wav("Samples/tom.wav")
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
for c in pulses:
  XX.append(c.start)
  XX.append(c.end)
  XX.append(None)
  YY.append(c.a)
  YY.append(c.a)
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

# Reconstruct pulses with maximun length
scaled_pulses = []
for p in pulses:
  FT = np.fft.rfft(p.W)
  normalized_W = np.fft.irfft(FT, max_length)
  scaled_pulses.append(Pulse(p.start, normalized_W))

# plot a reconstructed pulse
fig = go.Figure()

fig.update_layout(
    # width = 2 * np.pi * 220,
    # height = n * 220,
    # yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title="Original x Reconstructed",
    xaxis_title="Amplitude",
    yaxis_title="x",
    # font=dict(
    #     family="Courier New, monospace",
    #     size=18,
    #     color="#7f7f7f"
    # )
)

fig.add_trace(
  go.Scatter(
    # x=XX,
    y=pulses[31].normalized_W,
    # z=ZZ,
    name="Original",
    mode="lines+markers",
    # marker=dict(
    #     size=3,
    #     color=ZZ,
    #     colorscale='Viridis',
    #     showscale=True
    # )
  )
)

fig.add_trace(
  go.Scatter(
    # x=XX,
    y=scaled_pulses[31].normalized_W,
    # z=ZZ,
    name="Reconstructed",
    mode="lines+markers",
    # marker=dict(
    #     size=3,
    #     color=ZZ,
    #     colorscale='Viridis',
    #     showscale=True
    # )
  )
)

fig.show(config=dict({'scrollZoom': True}))

# fold, step = 1

scaled_pulses_W = []

for p in scaled_pulses:
  if p.a >= 0:
    scaled_pulses_W.append(p.normalized_W)
  else:
    scaled_pulses_W.append(-p.normalized_W)

scaled_pulses_W = np.array(scaled_pulses_W)

print("Step=1", np.average(np.var(scaled_pulses_W, 0)))

fig = go.Figure()

fig.update_layout(
    # width = 2 * np.pi * 220,
    # height = n * 220,
    # yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title="Step=1",
    xaxis_title="x",
    yaxis_title="amplitude",
    # font=dict(
    #     family="Courier New, monospace",
    #     size=18,
    #     color="#7f7f7f"
    # )
)

fig.add_trace(
  go.Scatter(
    x=np.tile(np.arange(max_length), len(scaled_pulses)),
    y=scaled_pulses_W.flat,
    # z=ZZ,
    name="Teste",
    mode="markers",
    # marker=dict(
    #     size=3,
    #     color=ZZ,
    #     colorscale='Viridis',
    #     showscale=True
    # )
  )
)

fig.show(config=dict({'scrollZoom': True}))

# fold, step = 2
for step in range(2, 5 + 1):
# step = 2
  nn = (scaled_pulses_W.size // (max_length * step)) * (max_length * step)
  A = np.reshape(scaled_pulses_W.flat[ : nn], (-1, max_length * step))

  print(f"Step={step}", np.average(np.var(A, 0)))

  # continue

  fig = go.Figure()

  fig.update_layout(
      # width = 2 * np.pi * 220,
      # height = n * 220,
      # yaxis = dict(scaleanchor = "x", scaleratio = 1),
      title=f"Step={step}",
      xaxis_title="x",
      yaxis_title="amplitude",
      # font=dict(
      #     family="Courier New, monospace",
      #     size=18,
      #     color="#7f7f7f"
      # )
  )

  fig.add_trace(
    go.Scatter(
      x=np.tile(np.arange(A.shape[1]), A.shape[0]),
      y=A.flat,
      # z=ZZ,
      name="Teste",
      mode="markers",
      # marker=dict(
      #     size=3,
      #     color=ZZ,
      #     colorscale='Viridis',
      #     showscale=True
      # )
    )
  )

  fig.show(config=dict({'scrollZoom': True}))
