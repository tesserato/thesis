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
fig.update_yaxes(tickvals=[i for i in range(n)])

fig.update_layout(
  width = np.pi * 440,
  height = n * 440,
  yaxis = dict(scaleanchor = "x", scaleratio = 1),
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
  m_vec = np.sqrt(x_vec**2 + y_vec**2)
  s = 1 if W[t] < 0 else 0
  s=0
  X_lines=[]
  Y_lines=[]
  ### vectors
  # X_lines.append(0)
  # Y_lines.append(0)
  # X_lines.append(x_vec)
  # Y_lines.append(y_vec)
  # X_lines.append(np.nan)
  # Y_lines.append(np.nan)

  ### isolines
  for k in np.arange(t+1):

    if t != 0:
      x0 = 0
      x1 = 2 * np.pi
      y0 = n * k / t - n * x0 / (2 * np.pi * t) + s * n / (2 * t)
      y1 = n * k / t - n * x1 / (2 * np.pi * t) + s * n / (2 * t)
    else:
      x0 = 2 * np.pi * k + s * np.pi
      x1 = 2 * np.pi * k + s * np.pi
      y0 = 0
      y1 = n
    X_lines.append(x0)
    Y_lines.append(y0)
    X_lines.append(x1)
    Y_lines.append(y1)
    X_lines.append(np.nan)
    Y_lines.append(np.nan)
  ### residues
  U = np.array([[x_vec], [y_vec]]) / m_vec
  P=np.array([[p], [f]])
  O=np.array([[s * np.pi], [0]])
  # d=np.sqrt((P-O).T @ U @ U.T @ (P-O))
  k_t = 2
  d =np.sqrt(  (n*(2*np.pi*k_t - p) - 2*np.pi*t*f)**2/(n**2 + 4*np.pi**2*t**2)  )
  # print(d)
  r = m_vec / 2 - np.sqrt((d - k * m_vec + m_vec/2)**2)

  fig.add_annotation( # TODO: put hovertext in the vector tip
  ax=p,
  ay=f,
  x= p - U[0, 0] * d, 
  y= f - U[1, 0] * d, 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=2,
  arrowsize=1,
  arrowwidth=2,
  hovertext= f"d {t}",
  arrowcolor="black")

  fig.add_annotation( # TODO: put hovertext in the vector tip
  ax=p,
  ay=f,
  x= p + U[0, 0] * d, 
  y= f + U[1, 0] * d, 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=2,
  arrowsize=1,
  arrowwidth=2,
  hovertext= f"d {t}",
  arrowcolor="black")

  # r = m_vec / 2 - np.sqrt((d - k * m_vec + m_vec/2)**2)
  # fig.add_annotation( # TODO: put hovertext in the vector tip
  # ax=p,
  # ay=f,
  # x= p - U[0, 0] * r[0][0], 
  # y= f - U[1, 0] * r[0][0], 
  # showarrow=True, 
  # text = "",
  # xref="x",
  # yref="y",
  # axref = "x", 
  # ayref = "y",
  # arrowhead=2,
  # arrowsize=1,
  # arrowwidth=2,
  # hovertext= f"r {t}",
  # arrowcolor="black")

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

#   fig.add_annotation( # TODO: put hovertext in the vector tip
#   ax=0, 
#   ay=0, 
#   x=U[0], 
#   y=U[1], 
#   showarrow=True, 
#   text = "",
#   xref="x",
#   yref="y",
#   axref = "x", 
#   ayref = "y",
#   arrowhead=2,
#   arrowsize=1,
#   arrowwidth=2,
#   hovertext= f"unit vector {t}",
#   arrowcolor="black"
# )



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



