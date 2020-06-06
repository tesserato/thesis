import plotly.graph_objects as go
import numpy as np


#######################
n = 1000

X = np.arange(n)

f = 10
p = np.pi
W = np.cos(p + 2 * np.pi * f * X / n)
#######################

fig = go.Figure()

fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
fig.update_yaxes(tickvals=[i for i in range(n)])

fig.update_layout(
  # width = 2000,
  height = 800,
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

avgposUx = 0
avgposUy = 0
avgnegUx = 0
avgnegUy = 0
for t in range(n):
  X_lines = []
  Y_lines = []
  den = np.sqrt(n**2 + 4*np.pi**2*t**2)
  Ux = n / den
  Uy = 2 * np.pi * t / den
  if W[t] >= 0:
    x0 = 0
    avgposUx += Ux * abs(W[t])
    avgposUy += Uy * abs(W[t])
  else:
    x0 = np.pi
    avgnegUx += Ux * abs(W[t])
    avgnegUy += Uy * abs(W[t])

  X_lines.append(x0)
  Y_lines.append(0)
  X_lines.append(x0 + Ux)
  Y_lines.append(Uy)

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

mpos = np.sqrt(avgposUx**2 + avgposUy**2)
fig.add_trace(
  go.Scatter(
    x=[0, avgposUx], 
    y=[0, avgposUy],
    hoverinfo=f"all",
    name=f"t={t}, a={W[t]:.2f}", 
    mode='lines',
    line=go.scatter.Line(
      width=1,
      dash="dash"
    )
  )
)

mneg = np.sqrt(avgnegUx**2 + avgnegUy**2)
fig.add_trace(
  go.Scatter(
    x=[np.pi, (np.pi + avgnegUx)],
    y=[0, avgnegUy],
    hoverinfo=f"all",
    name=f"t={t}, a={W[t]:.2f}", 
    mode='lines',
    line=go.scatter.Line(
      width=1,
      dash="dash"
    )
  )
)

print(round(avgposUx / avgposUy, 2), round(avgnegUx / avgnegUy, 2), n, f, p)

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



