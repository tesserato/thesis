import numpy as np
import plotly.graph_objects as go
import random

class region:
  def __init__(self, p0, p1, p2, p3, value=0):
    self.start = start
    self.end = end
    self.value = value

  def __str__(self):
    return f"[{self.start}, {self.end}]; v={round(self.value, 2)}"


### Generating random wave
n = 40
random.seed(1)
X = np.arange(n)
W = np.zeros(n)
number_of_random_waves = 5
A = np.array([random.uniform(1, 5) for i in range(number_of_random_waves)])
F = np.array([random.uniform(1, 5) for i in range(number_of_random_waves)])
P = np.array([random.uniform(0, 2 * np.pi) for i in range(number_of_random_waves)])
W = np.sum(A * np.cos(P + 2 * np.pi * F.T * X[:, np.newaxis] / n), 1)
W = W / np.max(np.abs(W))

fig = go.Figure()

# fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
# fig.update_yaxes(tickvals=[i for i in range(n)])

fig.update_layout(
    # width = 2 * np.pi * 220,
    # height = n * 300,
    # yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title=f"n = {n}",
    xaxis_title="Phase",
    yaxis_title="Frequency",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.add_trace(
  go.Scatter(
    x=[0, 0],
    y=[0, n//2+1],
    hoverinfo=f"all",
    name=f"t={0}, a={W[0]:.2f}", 
    mode='lines',
    line=go.scatter.Line(
      color="blue",
      width=10#max(1, W[x]**2 * 10)
    )
  )
)

x0 = 0
p0_regions = []
p0_fticks = []
p1_regions = []
p1_fticks = []

for x in range(1, n // 2 + 1):

  X_lines_e = []
  Y_lines_e = []
  X_lines_o = []
  Y_lines_o = []
  
  # w = 1 if W[x] >= 0 else -1
  w = W[x]

  for k in range(x // 2 + 1):
    p0_f0_e = np.round((4 * k - 1) * n / (4 * x), 2)
    p0_f1_e = np.round((4 * k + 1) * n / (4 * x), 2)
    p0_regions.append(interval(p0_f0_e, p0_f1_e, w))
    p0_f0_o = np.round((4 * k + 1) * n / (4 * x), 2)
    p0_f1_o = np.round((4 * k + 3) * n / (4 * x), 2)
    p0_regions.append(interval(p0_f0_o, p0_f1_o, -w))
    p0_fticks.append(p0_f0_e), p0_fticks.append(p0_f1_e), p0_fticks.append(p0_f0_o), p0_fticks.append(p0_f1_o)

    p1_f0_e = np.round((2 * k + 1) * n / (2 * x), 2)
    p1_f1_e = np.round((2 * k + 2) * n / (2 * x), 2)
    p1_regions.append(interval(p1_f0_e, p1_f1_e, w))
    p1_f0_o = np.round(2 * k * n / (2 * x), 2)
    p1_f1_o = np.round((2 * k + 1) * n / (2 * x), 2)
    p1_regions.append(interval(p1_f0_o, p1_f1_o, -w))
    p1_fticks.append(p1_f0_e), p1_fticks.append(p1_f1_e), p1_fticks.append(p1_f0_o), p1_fticks.append(p1_f1_o)

    X_lines_e.append(x0)
    X_lines_e.append(x0)
    X_lines_e.append(None)
    Y_lines_e.append(p0_f0_e)
    Y_lines_e.append(p0_f1_e)
    Y_lines_e.append(None)

    X_lines_o.append(x0)
    X_lines_o.append(x0)
    X_lines_o.append(None)
    Y_lines_o.append(p0_f0_o)
    Y_lines_o.append(p0_f1_o)
    Y_lines_o.append(None)

    X_lines_e.append(x0 + .25)
    X_lines_e.append(x0 + .25)
    X_lines_e.append(None)
    Y_lines_e.append(p1_f0_e)
    Y_lines_e.append(p1_f1_e)
    Y_lines_e.append(None)

    X_lines_o.append(x0 + .25)
    X_lines_o.append(x0 + .25)
    X_lines_o.append(None)
    Y_lines_o.append(p1_f0_o)
    Y_lines_o.append(p1_f1_o)
    Y_lines_o.append(None)

  x0 +=1

  fig.add_trace(
    go.Scatter(
      x=X_lines_e,
      y=Y_lines_e,
      hoverinfo=f"all",
      name=f"t={x}, a={W[x]:.2f}", 
      mode='lines',
      line=go.scatter.Line(
        color="red",
        width=10#max(1, W[x]**2 * 10)
      )
    )
  )

  fig.add_trace(
    go.Scatter(
      x=X_lines_o,
      y=Y_lines_o,
      hoverinfo=f"all",
      name=f"t={x}, a={W[x]:.2f}", 
      mode='lines',
      line=go.scatter.Line(
        color="blue",
        width=10#max(1, W[x]**2 * 10)
      )
    )
  )

fticks = p0_fticks + p1_fticks
fticks = np.unique(np.array(fticks))

# print(len(p0_fticks))
# p0_fticks = np.unique(np.array(p0_fticks))
# print(p0_fticks.shape)

# print(len(p1_fticks))
# p1_fticks = np.unique(np.array(p1_fticks))
# print(p1_fticks.shape)


p0_summation = [interval(fticks[i], fticks[i + 1], 0) for i in range(len(fticks) - 1)]
p1_summation = [interval(fticks[i], fticks[i + 1], 0) for i in range(len(fticks) - 1)]

for r in p0_regions:
  for s in p0_summation:
    if r.start <= s.start and r.end >= s.end:
      s.value += r.value * 2 / n

for r in p1_regions:
  for s in p1_summation:
    if r.start <= s.start and r.end >= s.end:
      s.value += r.value * 2 / n

for s in p0_summation:
  fig.add_trace(
    go.Scatter(
      x=[x0, x0],
      y=[s.start, s.end],
      hoverinfo=f"all",
      name=f"t={x}, a={s.value:.2f}", 
      mode='lines',
      line=go.scatter.Line(
        color= "red" if s.value >= 0 else "blue",
        width= max(abs(s.value) * 10, 4)
      )
    )
  )

for s in p1_summation:
  fig.add_trace(
    go.Scatter(
      x=[x0 + 1, x0 + 1],
      y=[s.start, s.end],
      hoverinfo=f"all",
      name=f"t={x}, a={s.value:.2f}", 
      mode='lines',
      line=go.scatter.Line(
        color= "red" if s.value >= 0 else "blue",
        width= max(abs(s.value) * 10, 4)
      )
    )
  )

fig.update_yaxes(tickvals=fticks)

# fig.show(config=dict({'scrollZoom': False}))

###################################
###################################
###################################


idx = np.argmax(A)

p0_max = interval(0,0,0)
p1_max = interval(0,0,0)
gl_max = interval(0,0,0)

for i in range(len(p0_summation)):
  if abs(p0_summation[i].value) > abs(p0_max.value):
    p0_max = p0_summation[i]

  if abs(p1_summation[i].value) > abs(p1_max.value):
    p1_max = p1_summation[i]

  m = np.sqrt(p0_summation[i].value**2 + p1_summation[i].value**2)
  if m > gl_max.value:
    gl_max = interval(p0_summation[i].start, p0_summation[i].end, m)

idx = np.argmax(A)

FT = np.fft.rfft(W) * 2 / n

f = np.argmax(np.abs(FT))


print(round(F[idx], 2), round(A[idx], 2), p0_max, p1_max, gl_max, f, np.abs(FT[f]))

fig = go.Figure()

# fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
# fig.update_yaxes(tickvals=[i for i in range(n)])

fig.update_layout(
  # width = 2 * np.pi * 220,
  # height = n * 300,
  # yaxis = dict(scaleanchor = "x", scaleratio = 1),
  title=f"n = {n}",
  xaxis_title="Phase",
  yaxis_title="Frequency",
  font=dict(
      family="Courier New, monospace",
      size=18,
      color="#7f7f7f"
  )
)

idxs = np.argsort(F)

fig.add_trace(
  go.Scatter(
    x=F[idxs],
    y=A[idxs],
    hoverinfo=f"all",
    name=f"t={0}, a={W[0]:.2f}", 
    mode='lines',
    line=go.scatter.Line(
      color="black",
      width=3
    )
  )
)

fig.add_trace(
  go.Scatter(
    x=X,
    y=np.abs(FT.real),
    hoverinfo=f"all",
    name=f"t={0}, a={W[0]:.2f}", 
    mode='lines',
    line=go.scatter.Line(
      color="red",
      width=3
    )
  )
)

fig.add_trace(
  go.Scatter(
    x=X,
    y=np.abs(FT.imag),
    hoverinfo=f"all",
    name=f"t={0}, a={W[0]:.2f}", 
    mode='lines',
    line=go.scatter.Line(
      color="blue",
      width=3
    )
  )
)

fig.add_trace(
  go.Scatter(
    x=np.array([[r.start, r.end] for r in p0_summation]).flat,
    y=np.abs(np.array([[r.value, r.value] for r in p0_summation]).flat),
    hoverinfo=f"all",
    name=f"t={0}, a={W[0]:.2f}", 
    mode='lines',
    line=go.scatter.Line(
      color="red",
      width=3
    )
  )
)

fig.add_trace(
  go.Scatter(
    x=np.array([[r.start, r.end] for r in p1_summation]).flat,
    y=np.abs(np.array([[r.value, r.value] for r in p1_summation]).flat),
    hoverinfo=f"all",
    name=f"t={0}, a={W[0]:.2f}", 
    mode='lines',
    line=go.scatter.Line(
      color="blue",
      width=3
    )
  )
)

fig.update_xaxes(range=[1, 7])

fig.show(config=dict({'scrollZoom': False}))