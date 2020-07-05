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
    self.n = W.shape[0]
    self.W = W
    self.normalized_W = W / np.max(np.abs(W))
    self.a = np.average(W)
    self.group = None


W, fps = read_wav("Samples/piano33.wav")
W = W - np.average(W)
a = np.max(np.abs(W))
W = W / a

# W = savgol_filter(W, 5, 3)

W = W[ 70000: 80000]

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

# # Reconstruct pulses with maximun length
# scaled_pulses = []
# for p in pulses:
#   FT = np.fft.rfft(p.W)
#   normalized_W = np.abs(np.fft.irfft(FT, max_length))
#   if p.end - p.start > 1:
#     scaled_pulses.append(Pulse(p.start, normalized_W))
# print(len(scaled_pulses))

print(len(pulses))
pulses = [p for p in pulses if p.n > 1]
print(len(pulses))

correlations = np.ones((len(pulses), len(pulses)))
# correlations.fill(999)
for i in range(len(pulses)):
  a1 = np.abs(pulses[i].a)
  n1 = pulses[i].n
  for j in range(i+1, len(pulses)):
    a2 = np.abs(pulses[j].a)
    n2 = pulses[j].n
    cor =( np.abs(a1 - a2) / max([a1, a2]) + np.abs(n1 - n2) / max([n1, n2]) ) / 2
    correlations[i, j] = cor

fig = go.Figure()

fig.add_trace(
  go.Heatmap(
    z=correlations,
    name=f"Half Cycles",
    colorscale='Viridis'
    # marker=dict(
    #     size=8,
    #     color="black", #set color equal to a variable
    #     showscale=False
    # )
  )
)

fig.show(config=dict({'scrollZoom': True}))
# for i in range(len(pulses)):
#   correlations[i, j] = 999
# groups = 0
# while np.min(correlations) < 999:
for i in range(50):
  idx1, idx2 = np.unravel_index(correlations.argmin(), correlations.shape)
  print(correlations[idx1, idx2], idx1, idx2)
  correlations[idx1, idx2] = 999
  correlations[idx2, idx1] = 999
#   if scaled_pulses[idx1].group == None and scaled_pulses[idx2].group == None:
#     scaled_pulses[idx1].group = groups
#     scaled_pulses[idx2].group = groups
#     groups += 1
#   else:
#     if scaled_pulses[idx1].group == None:
#       scaled_pulses[idx1].group = scaled_pulses[idx2].group
#     else:
#       scaled_pulses[idx2].group = scaled_pulses[idx1].group  

# print(len(scaled_pulses), groups)
# fig = go.Figure()

# fig.update_layout(
#     # width = 2 * np.pi * 220,
#     # height = n * 220,
#     # yaxis = dict(scaleanchor = "x", scaleratio = 1),
#     title=f"Cycles",
#     xaxis_title="x",
#     yaxis_title="Amplitude",
#     # font=dict(
#     #     family="Courier New, monospace",
#     #     size=18,
#     #     color="#7f7f7f"
#     # )
# )

# fig.add_trace(
#   go.Scatter(
#     # x=X,
#     y=scaled_pulses[idx1].normalized_W,
#     name="Signal W[x]",
#     mode="markers",
#     # marker=dict(
#     #     size=8,
#     #     color="red", #set color equal to a variable
#     #     showscale=False
#     # )
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     # x=XX,
#     y=scaled_pulses[idx2].normalized_W,
#     name=f"Half Cycles",
#     mode="markers",
#     # marker=dict(
#     #     size=8,
#     #     color="black", #set color equal to a variable
#     #     showscale=False
#     # )
#   )
# )

# fig.show(config=dict({'scrollZoom': True}))