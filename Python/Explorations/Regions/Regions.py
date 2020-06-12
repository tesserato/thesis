import numpy as np
import plotly.graph_objects as go
import random

class interval:
  def __init__(self, start, end, value=0):
    self.start = start
    self.end = end
    self.value = value



### Generating random wave
n = 20
random.seed(0)
X = np.arange(n)
W = np.zeros(n)
number_of_random_waves = 5
A = np.array([random.uniform(1, 5) for i in range(number_of_random_waves)])
F = np.array([random.uniform(1, 9) for i in range(number_of_random_waves)])
P = np.array([random.uniform(0, 2 * np.pi) for i in range(number_of_random_waves)])
W = np.sum(A * np.cos(P + 2 * np.pi * F.T * X[:, np.newaxis] / n), 1)
W = W / np.max(np.abs(W))
idx = np.argmax(A)

print(F[idx])

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
regions = [] # list of [start, end, weight] #[[0, 1 if W[0] >= 0 else 0], [n//2+1, 1 if W[0] >= 0 else 0]]
yticks = []
for x in range(1, n // 2 + 1):

  X_lines_e = []
  Y_lines_e = []
  X_lines_o = []
  Y_lines_o = []
  
  w = 1 if W[x] >= 0 else -1

  for k in range(x // 2 + 1):
    f0_e = np.round((4 * k - 1) * n / (4 * x), 2)
    f1_e = np.round((4 * k + 1) * n / (4 * x), 2)
    regions.append(interval(f0_e, f1_e, w))
    f0_o = np.round((4 * k + 1) * n / (4 * x), 2)
    f1_o = np.round((4 * k + 3) * n / (4 * x), 2)
    regions.append(interval(f0_o, f1_o, -w))
    yticks.append(f0_e), yticks.append(f1_e), yticks.append(f0_o), yticks.append(f1_o)

    X_lines_e.append(x0)
    X_lines_e.append(x0)
    X_lines_e.append(None)
    Y_lines_e.append(f0_e)
    Y_lines_e.append(f1_e)
    Y_lines_e.append(None)

    X_lines_o.append(x0)
    X_lines_o.append(x0)
    X_lines_o.append(None)
    Y_lines_o.append(f0_o)
    Y_lines_o.append(f1_o)
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

print(len(yticks))
yticks = np.unique(np.array(yticks))
print(yticks.shape)

summation = [interval(yticks[i], yticks[i + 1], 0) for i in range(len(yticks) - 1)]

for r in regions:
  for s in summation:
    if r.start <= s.start and r.end >= s.end:
      s.value += r.value

for s in summation:
  fig.add_trace(
    go.Scatter(
      x=[x0, x0],
      y=[s.start, s.end],
      hoverinfo=f"all",
      name=f"t={x}, a={s.value:.2f}", 
      mode='lines',
      line=go.scatter.Line(
        color="red",
        width= abs(s.value) * 5
      )
    )
  )

fig.update_yaxes(tickvals=yticks)

fig.show(config=dict({'scrollZoom': False}))