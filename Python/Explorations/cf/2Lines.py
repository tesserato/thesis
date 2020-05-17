import plotly.graph_objects as go
import numpy as np

r = np.array([[0, -1], [1, 0]])  # rotation matrix

p1 = np.array([[4], [2]])        # point in l1
v1 = np.array([[1], [3]])        # vector perpendicular to l1
n1 = v1 / np.sqrt(v1.T @ v1)     # unit vector perpendicular to l1
u1 = r @ n1                      # unit vector in the direction of l1

p2 = np.array([[8], [-1]])        # point in l2
v2 = np.array([[1], [2]])        # vector perpendicular to l2
n2 = v2 / np.sqrt(v2.T @ v2)     # unit vector perpendicular to l2
u2 = r @ n2                      # unit vector in the direction of l2

xi = -9
xf = 9
l1i = (xi - p1[0, 0]) / u1[0, 0]
l1f = (xf - p1[0, 0]) / u1[0, 0]
y1i = (p1 + l1i * u1)[1, 0]
y1f = (p1 + l1f * u1)[1, 0]

l2i = (xi - p2[0, 0]) / u2[0, 0]
l2f = (xf - p2[0, 0]) / u2[0, 0]
y2i = (p2 + l2i * u2)[1, 0]
y2f = (p2 + l2f * u2)[1, 0]

p = np.array([[4], [6]])

d1 = np.sqrt((p - p1).T @ n1 @ n1.T @ (p - p1))
d2 = np.sqrt((p - p2).T @ n2 @ n2.T @ (p - p2))


min_x = min(xi, xf)
max_x = max(xi, xf)
min_y = int(np.floor(min([y1i, y1f, y2i, y2f])))
max_y = int(np.ceil(max([y1i, y1f, y2i, y2f])))

fig = go.Figure()

fig.update_xaxes(tickvals=[i for i in range(min_x, max_x + 1, 1)])
fig.update_yaxes(tickvals=[i for i in range(min_y, max_y + 1, 1)])

##### l1
fig.add_trace(
  go.Scatter(
    name="line 1",
    x=[xi, xf], 
    y=[y1i, y1f],
    line = dict(color='blue'),
    mode='lines',
  ))
##### d1
fig.add_trace(
  go.Scatter(
    name="d 1",
    x=[p[0,0], p[0,0] - n1[0,0] * d1[0][0]], 
    y=[p[1,0], p[1,0] - n1[1,0] * d1[0][0]],
    line = dict(color='blue', dash="dash"),
    mode='lines',
  ))

##### l2
fig.add_trace(
  go.Scatter(
    name="line 2",
    x=[xi, xf], 
    y=[y2i, y2f],
    line = dict(color='red'),
    mode='lines',))
##### d2
fig.add_trace(
  go.Scatter(
    name="d 2",
    x=[p[0,0], p[0,0] - n2[0,0] * d2[0][0]], 
    y=[p[1,0], p[1,0] - n2[1,0] * d2[0][0]],
    line = dict(color='red', dash="dash"),
    mode='lines',))

wdt = 2
##### v1
fig.add_annotation( 
  ax=0,
  ay=0, 
  x=v1[0, 0], 
  y=v1[1, 0], 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=int(wdt/2),
  arrowsize=wdt,
  arrowwidth=wdt,
  hovertext= f"v 1",
  arrowcolor="blue")
##### n1
fig.add_annotation( 
  ax=0,
  ay=0, 
  x=n1[0, 0], 
  y=n1[1, 0], 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=int(wdt/2),
  arrowsize=wdt,
  arrowwidth=wdt,
  hovertext= f"n 1",
  arrowcolor="blue")
##### u1
fig.add_annotation( 
  ax=0,
  ay=0, 
  x=u1[0, 0], 
  y=u1[1, 0], 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=int(wdt/2),
  arrowsize=wdt,
  arrowwidth=wdt,
  hovertext= f"u 1",
  arrowcolor="blue")

##### v2
fig.add_annotation( 
  ax=0,
  ay=0, 
  x=v2[0, 0], 
  y=v2[1, 0], 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=int(wdt/2),
  arrowsize=wdt,
  arrowwidth=wdt,
  hovertext= f"v 1",
  arrowcolor="red")
##### n2
fig.add_annotation( 
  ax=0,
  ay=0, 
  x=n2[0, 0], 
  y=n2[1, 0], 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=int(wdt/2),
  arrowsize=wdt,
  arrowwidth=wdt,
  hovertext= f"n 1",
  arrowcolor="red")
##### u2
fig.add_annotation( 
  ax=0,
  ay=0, 
  x=u2[0, 0], 
  y=u2[1, 0], 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=int(wdt/2),
  arrowsize=wdt,
  arrowwidth=wdt,
  hovertext= f"u 2",
  arrowcolor="red")

##### p1
fig.add_trace(
  go.Scatter(
    name="point",
    x=[p1[0, 0]], 
    y=[p1[1, 0]],
    mode='markers',
    marker=dict(
        size=8,
        color="blue", #set color equal to a variable
        showscale=False
  )))
##### p2
fig.add_trace(
  go.Scatter(
    name="point",
    x=[p2[0, 0]], 
    y=[p2[1, 0]],
    mode='markers',
    marker=dict(
        size=8,
        color="red", #set color equal to a variable
        showscale=False
  )))
##### p
fig.add_trace(
  go.Scatter(
    name="point",
    x=[p[0, 0]], 
    y=[p[1, 0]],
    mode='markers',
    marker=dict(
        size=8,
        color="black", #set color equal to a variable
        showscale=False
  )))

fig.update_layout(
  width = (max_x - min_x) * 80,
  height = (max_y - min_y) * 80,
  yaxis = dict(scaleanchor = "x", scaleratio = 1),
  # title=f"n = {n}",
  # xaxis_title="Phase",
  # yaxis_title="Frequency",
  font=dict(
      family="Courier New, monospace",
      size=18,
      color="#7f7f7f"
  ))

fig.show(config=dict({'scrollZoom': False}))