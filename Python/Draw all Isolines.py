import plotly.graph_objects as go
import numpy as np


#######################
n = 10

X = np.arange(n)

f = 3
p = np.pi / 2
W = np.cos(p + 2 * np.pi * f * X / n)
#######################

fig = go.Figure()

fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
fig.update_yaxes(tickvals=[i for i in range(n)])

fig.update_layout(
    # width = 2 * np.pi * 220,
    height = n * 220,
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

for t in range(n):
  X_lines=[]
  Y_lines=[]
  for k in range(1, t + 1):
    f0 = n * k / t
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
      hoverinfo=f"all",
      name=f"t={t}, a={W[t]:.2f}", 
      mode='lines',
      line=go.scatter.Line(
        width=max(1, W[t]**2 * 10),
        dash="solid" if W[t]>0 else "dash"
        )
    )
  )

fig.add_trace(
  go.Scatter(
    x=[p], 
    y=[f],
    name=f"max", 
    mode='markers',
    marker=dict(
        size=8,
        color="black", #set color equal to a variable
        showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': False}))



