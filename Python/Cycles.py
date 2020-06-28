import plotly.graph_objects as go
import random
import numpy as np
from Helper import read_wav
from scipy.signal import savgol_filter

class Cycle:
  cycles = []
  max_n = 0
  def __init__(self, x0, W):
    Cycle.cycles.append(self)
    self.start = x0 - .5
    self.end = self.start + W.shape[0]
    self.n = self.end - self.start
    self.W = W
    self.a = np.average(W)
    centered_W = W - np.average(W)
    self.f = np.argmax(np.abs(np.fft.rfft(centered_W)))
    if self.n > Cycle.max_n:
      Cycle.max_n = self.n
    sign = np.sign(W[0])
    self.zeros = 0
    for w in centered_W:
      if sign != np.sign(w):
        self.zeros += 1
        sign = np.sign(w)


W, fps = read_wav("Samples/tom.wav")
W = W - np.average(W)
a = np.max(np.abs(W))
W = W / a

W = savgol_filter(W, 5, 3)

# W = W [ : 1000]

n = W.shape[0]
X = np.arange(n)


sign = np.sign(W[0])
x0 = 0
for x in range(n):
  if sign != np.sign(W[x]):
    Cycle(x0, W[x0 : x])
    x0 = x
    sign = np.sign(W[x])
Cycle(x0, W[x0 : n])

XX = []
YY = []
for c in Cycle.cycles:
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
    name="W[x]",
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

#######################################
#######################################
XX = []
YY = []
ZZ = []
for i, c in enumerate(Cycle.cycles):
  XX.append(c.a)
  YY.append(c.zeros)
  ZZ.append(i)

fig = go.Figure()

fig.update_layout(
    # width = 2 * np.pi * 220,
    # height = n * 220,
    # yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title="Length x Amplitude",
    xaxis_title="Amplitude",
    yaxis_title="Zeros",
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
    # z=ZZ,
    name="Length x Amplitude",
    mode="markers",
    marker=dict(
        size=3,
        color=ZZ,
        colorscale='Viridis',
        showscale=True
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))
exit()
#######################################
#######################################
XX = []
YY = []
for i in range(1, len(Cycle.cycles)):
  XX.append(Cycle.cycles[i - 1].f)
  YY.append(Cycle.cycles[i].f)

fig = go.Figure()

fig.update_layout(
    # width = 2 * np.pi * 220,
    # height = n * 220,
    # yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title="f(x) x f(x+1)",
    xaxis_title="f(x)",
    yaxis_title="f(x+1)",
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
    # z=ZZ,
    name="f(x) x f(x+1)",
    mode="lines+markers",
    marker=dict(
        size=3,
        color=ZZ,
        colorscale='Viridis',
        showscale=True
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))

#######################################
#######################################

XX = np.zeros(int(Cycle.max_n // 2 + 1))

for c in Cycle.cycles:
  XX[c.f] += 1

XX = XX / len(Cycle.cycles)

fig = go.Figure()

fig.update_layout(
    # width = 2 * np.pi * 220,
    # height = n * 220,
    # yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title="f(x) x f(x+1)",
    xaxis_title="f(x)",
    yaxis_title="f(x+1)",
    # font=dict(
    #     family="Courier New, monospace",
    #     size=18,
    #     color="#7f7f7f"
    # )
)

fig.add_trace(
  go.Scatter(
    # x=XX,
    y=XX,
    # z=ZZ,
    name="f(x) x f(x+1)",
    mode="lines+markers",
    marker=dict(
        size=3,
        color=ZZ,
        colorscale='Viridis',
        showscale=True
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))

