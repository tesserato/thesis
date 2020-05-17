# import plotly.express as px
import plotly.graph_objects as go
import numpy as np


#######################
n = 10
X = np.arange(n)
f = 1
p = np.pi / 2
W = np.cos(p + 2 * np.pi * f * X / n)
#######################

fig = go.Figure()
fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
fig.update_yaxes(tickvals=[i for i in range(-n, n+1)])

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
    )))

fig.update_layout(
  width = (2 * np.pi) * 220,
  height = (2 * n + 1) * 220,
  # yaxis = dict(scaleanchor = "x", scaleratio = 1),
  title=f"n = {n}",
  xaxis_title="Phase",
  yaxis_title="Frequency",
  font=dict(
      family="Courier New, monospace",
      size=18,
      color="#7f7f7f"
  ))

for t in range(n):
  x_vec = (2 * np.pi * n**2) / (n**2 + 4 * np.pi**2 * t**2)     # p
  y_vec = (4 * np.pi**2 * n * t) / (n**2 + 4 * np.pi**2 * t**2) # f
  V = np.array([[x_vec], [y_vec]])
  m_vec = np.sqrt(x_vec**2 + y_vec**2)
  R = np.array([[0, -1], [1, 0]])
  U = R @ V #/ m_vec

  # fig.add_annotation( # TODO: put hovertext in the vector tip
  # ax=0, 
  # ay=0, 
  # x=V[0, 0], 
  # y=V[1, 0], 
  # showarrow=True, 
  # text = "",
  # xref="x",
  # yref="y",
  # axref = "x", 
  # ayref = "y",
  # arrowhead=2,
  # arrowsize=1,
  # arrowwidth=2,
  # hovertext= f"unit vector {t}",
  # arrowcolor="black")

  k_t = 2
  d =np.sqrt(  (n*(2*np.pi*k_t - p) - 2*np.pi*t*f)**2/(n**2 + 4*np.pi**2*t**2)  )
  fig.add_annotation( # TODO: put hovertext in the vector tip
  ax=p, 
  ay=f, 
  x=p + d * V[0, 0] / m_vec, 
  y=f + d * V[1, 0] / m_vec, 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=2,
  arrowsize=1,
  arrowwidth=2,
  hovertext= f"unit vector {t}",
  arrowcolor="black")

  X_lines=[]
  Y_lines=[]
  for k in range(t+1):
    P0 = np.array([[k * 2 * np.pi], [0]]) # XY0 = P0 + lambda * U
    l0 = (-np.pi - P0[0, 0]) / U[0, 0]
    XY0 = P0 + l0 * U
    l1 = (3 * np.pi - P0[0, 0]) / U[0, 0]
    XY1 = P0 + l1 * U
    X_lines.append(XY0[0, 0])
    Y_lines.append(XY0[1, 0])
    X_lines.append(XY1[0, 0])
    Y_lines.append(XY1[1, 0])
    X_lines.append(np.nan)
    Y_lines.append(np.nan)

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
    ))



fig.show(config=dict({'scrollZoom': False}))



