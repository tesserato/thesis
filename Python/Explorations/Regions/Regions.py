import numpy as np
import plotly.graph_objects as go
import random

### Generating random wave
n = 10
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
    y=[-n, n],
    hoverinfo=f"all",
    name=f"t={0}, a={W[0]:.2f}", 
    mode='lines',
    line=go.scatter.Line(
      color="blue",
      width=10#max(1, W[x]**2 * 10)
    )
  )
)

regions0 = [[-n, n, 1 if W[0] >= 0 else 0]]
regions1 = []
x0 = 1
yticks = []
for x in range(1, n // 2 + 1):

  X_lines_e = []
  Y_lines_e = []
  X_lines_o = []
  Y_lines_o = []
  
  w = 1 if W[x] >= 0 else 0
  regions1 = []
  for k in range(x // 2 + 1):
    f0_e = np.round((4 * k - 1) * n / (4 * x), 2)
    f1_e = np.round((4 * k + 1) * n / (4 * x), 2)

    f0_o = np.round((4 * k + 1) * n / (4 * x), 2)
    f1_o = np.round((4 * k + 3) * n / (4 * x), 2)

    for r in regions0:
      if r[0] < f0_e:
        if r[1] > f1_e: # r contains f
          print("r contains fe")
          regions1.append([f0_e, f1_e, r[2] + 1])
          # regions1.append([f1_e, r[1], 1])
        # else: # r1 < f1
        #   print("r[0] < f0 and r1 < f1")
        #   regions1.append([r[0], f0_e, 1])
        #   regions1.append([f0_e, r[1], r[2] + 1])
        #   regions1.append([r[1], f1_e, r[2] + 1])   

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
    # yticks.append(f0)
    # yticks.append(f1)
    
  for l in regions1:
    fig.add_trace(
      go.Scatter(
        x=[.5, .5],
        y=[l[0], l[1]],
        hoverinfo=f"all",
        name=f"t={x}, a={l[2]:.2f}", 
        mode='lines',
        line=go.scatter.Line(
          # color="black",
          width=10#max(1, W[x]**2 * 10)
        )
      )
    )
  regions0[:] = regions1[:]
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

# print(len(yticks))
# yticks = np.unique(np.array(yticks))
# print(yticks.shape)

fig.update_yaxes(tickvals=yticks)

fig.show(config=dict({'scrollZoom': False}))