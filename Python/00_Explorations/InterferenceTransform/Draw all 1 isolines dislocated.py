import plotly.graph_objects as go
import numpy as np


#######################
n = 10

X = np.arange(n)

f = 1
p = np.pi
W = np.cos(p + 2 * np.pi * f * X / n)
#######################

fig = go.Figure()

fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], range=[0, 2*np.pi])
fig.update_yaxes(tickvals=[i for i in range(n)], range=[0, n / 2])

fig.update_layout(
  # width = 2 * np.pi * 220,
  height = n * 220,
  # xaxis = dict(),
  yaxis = dict(scaleanchor = "x", scaleratio = 1),
  title=f"n = {n}",
  xaxis_title="Phase",
  yaxis_title="Frequency",
  xaxis={"mirror" : "allticks", "side": "top"},
  # yaxis={"mirror" : , 'side': 'right'},
  font=dict(
    family="Courier New, monospace",
    size=18,
    color="#7f7f7f"
  )
)

for t in range(n):
  X_lines=[]
  Y_lines=[]
  for k in range(t + 1):
    if W[t] >= 0:
      xi = 2 * np.pi * k
      yi = 0
      xf = 2 * np.pi * k - 2 * np.pi * t
      yf = n
    else:
      xi = 2 * np.pi * k + np.pi
      yi = 0
      xf = 2 * np.pi * k + np.pi - 2 * np.pi * t
      yf = n

    X_lines.append(xi), Y_lines.append(yi)
    X_lines.append(xf), Y_lines.append(yf)
    X_lines.append(None), Y_lines.append(None)

  fig.add_trace(
    go.Scatter(
      x=X_lines,
      y=Y_lines,
      hoverinfo=f"all",
      name=f"t={t}, a={W[t]:.2f}", 
      mode='lines',
      line=go.scatter.Line(
        width=max(1, abs(W[t]) * 10)
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



