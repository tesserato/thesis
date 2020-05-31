import plotly.express as px
import plotly.graph_objects as go
import numpy as np


#######################
n = 10
res_f = 10
res_p = 10
min_f = 0
max_f = n
#######################

fig = go.Figure()

fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])

fig.update_layout(
    width = 2 * np.pi * 220,
    height = max_f * 220,
    yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title=f"n = {n}",
    xaxis_title="Phase",
    yaxis_title="Frequency",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

for t in range(1, n + 1):
  X_lines=[]
  Y_lines=[]
  for k in range(1, t + 1):
    f0 = n * k / t #- n * p / (2 * np.pi * t)
    f2pi = n * k / t - n * 2 * np.pi / (2 * np.pi * t)
    X_lines.append(0)
    X_lines.append(2 * np.pi)
    X_lines.append(np.nan)
    Y_lines.append(f0)
    Y_lines.append(f2pi)
    Y_lines.append(np.nan)

  p_vec = (2 * np.pi * n**2) / (n**2 + 4 * np.pi**2 * t**2)
  f_vec = (4 * np.pi**2 * n * t) / (n**2 + 4 * np.pi**2 * t**2)

  X_lines.append(0)
  Y_lines.append(0)
  X_lines.append(p_vec)
  Y_lines.append(f_vec)

  fig.add_trace(
    go.Scatter(
      x=X_lines, 
      y=Y_lines,
      name=f"t={t}", 
      mode='lines',
      # line=go.scatter.Line(color="red")
    )
  )

fig.show(config=dict({'scrollZoom': False}))



