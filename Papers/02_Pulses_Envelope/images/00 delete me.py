# import plotly.express as px
import plotly.graph_objects as go
import numpy as np


#######################
n = 7
t = 3
res_f = 10
res_p = 10
min_f = 0
max_f = n
#######################



F = np.linspace(0, max_f, res_f)
P = np.linspace(0, 2 * np.pi, res_p)

FP = np.zeros((res_f, res_p))

for i in range(res_f):
  for j in range(res_p):
    FP[i,j] += np.cos(P[j] + 2*np.pi*F[i]*t/n)


X, Y, Z = [], [], []
for i in range(res_f):
  for j in range(res_p):
    X.append(P[j])
    Y.append(F[i])
    Z.append(FP[i,j])
    
    
fig = go.Figure(data=go.Scatter(
    x= X,
    y= Y,
    name= "Amplitudes",
    legendgroup="Normal",
    customdata=Z,
    hovertemplate='phase:%{x:.3f}<br>frequency:%{y:.3f}<br>amplitude: %{customdata:.3f} ',
    mode='markers',
    marker=dict(
        size=18,
        color=Z, #set color equal to a variable
        colorscale='Bluered', # one of plotly colorscales
        showscale=False
    )
))

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

fig.add_trace(go.Scatter(x=X_lines, y=Y_lines,name=f"Amplitude=1", mode='lines',line=go.scatter.Line(color="red")))

p_vec = (2 * np.pi * n**2) / (n**2 + 4 * np.pi**2 * t**2)
f_vec = (4 * np.pi**2 * n * t) / (n**2 + 4 * np.pi**2 * t**2)

fig.add_annotation(
  ax=0, 
  ay=0, 
  x=p_vec, 
  y=f_vec, 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=2,
  arrowsize=1,
  arrowwidth=2,
  hovertext= "original vector",
  arrowcolor="black"
)

fig.add_annotation(
  ax=0, 
  ay=0, 
  x=0, 
  y=np.sqrt(f_vec**2 + p_vec**2), 
  showarrow=True, 
  text = "",
  xref="x",
  yref="y",
  axref = "x", 
  ayref = "y",
  arrowhead=2,
  arrowsize=1,
  arrowwidth=2,
  hovertext= "vertical vector"
)

links_X = []
links_Y = []
for i in range(len(Z)):
  for j in range(i, len(Z)):
    if round(Z[i], 3) == round(Z[j], 3) and X[i] != X[j] and Y[i] != Y[j]:
      links_X.append(X[i])
      links_X.append(X[j])
      links_X.append(np.nan)
      links_Y.append(Y[i])
      links_Y.append(Y[j])
      links_Y.append(np.nan)
fig.add_trace(go.Scatter(x=links_X, y=links_Y, name=f"Links", opacity=0.8, mode='lines',line=go.scatter.Line(color="gray")))

# all_amps = np.unique(np.round(np.array(Z), 3))

# print(all_amps.shape[0])

##########################################
theta = np.arctan(p_vec / f_vec)
R_cw  = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
R_ccw = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
X, Y = R_ccw @ np.array([X, Y])
fig.add_trace(go.Scatter(
    x= X,
    y= Y,
    name= "Amplitudes",
    legendgroup="Rotated",
    customdata=Z,    
    hovertemplate='phase:%{x:.3f}<br>frequency:%{y:.3f}<br>amplitude: %{customdata:.3f} ',
    mode='markers',
    marker=dict(
        size=8,
        color=Z, #set color equal to a variable
        colorscale='Bluered', # one of plotly colorscales
        showscale=False
    )
))

X_lines, Y_lines = R_ccw @ np.array([X_lines, Y_lines])
min_x = np.nanmin(X_lines)
X_lines = np.tile(np.array([min_x, 2 * np.pi, np.nan]), X_lines.shape[0]//3)
# print(X_lines)
fig.add_trace(go.Scatter(x=X_lines, y=Y_lines,name=f"Amplitude=1", legendgroup="Rotated", mode='lines',line=go.scatter.Line(color="red")))

grid_Y=[]
grid_X=[]
for y in Y:
  grid_X.append(min_x)
  grid_X.append(2 * np.pi)
  grid_X.append(np.nan)
  grid_Y.append(y)
  grid_Y.append(y)
  grid_Y.append(np.nan)

fig.add_trace(go.Scatter(x=grid_X, y=grid_Y,name=f"Grid", opacity=0.3, legendgroup="Rotated", mode='lines',line=go.scatter.Line(color="gray")))

##########################################
fig.update_xaxes(tickvals=P)
fig.update_yaxes(tickvals=F)

# for t in range(n, n+1):


fig.update_layout(
    width = 2 * np.pi * 200,
    height = max_f * 200,
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

config = dict({'scrollZoom': False})
fig.show(config=config)


###########################################################
###########################################################

import numpy as np
import random
import plotly.graph_objects as go


random.seed(0)
n = 1000+1

X = np.arange(n)
Y = np.zeros(n)

for _ in range(100):
  a = random.uniform(1, 5)
  f = random.uniform(1, 10)
  p = random.uniform(0, 2 * np.pi)
  Y += a * np.cos(p + 2 * np.pi * f * X / n)

Y = Y / np.max(np.abs(Y))

X_lowres = X[::20]
Y_lowres = np.round(Y[::20], 1)

fig = go.Figure()

fig.layout.template ="plotly_white"

fig.update_layout(
    title="Continuous x Discrete Wave",
    xaxis_title="Time",
    yaxis_title="Amplitude",
    # font=dict(
    #     family="Courier New, monospace",
    #     size=18,
    #     color="#7f7f7f"
    # )
)

fig.update_yaxes(tickvals=np.linspace(-1, 1, 21), zerolinewidth=2, zerolinecolor='black')

fig.update_layout(legend_orientation="h")

# fig.update_xaxes()

# fig.update_layout(
#     yaxis = dict(
#         # tickmode = 'linear',
#         # tick0 = 0.5,
#         dtick = 0.1
#     )
# )

fig.add_trace(go.Scatter(name= "Original Wave",
  x=X,
  y=Y,
  mode='lines',
  line_shape='spline',
  line=go.scatter.Line(color="blue", width=2)
  ))

fig.add_trace(go.Scatter(name= "Discrete Wave Stem",
  # legendgroup="Discrete Wave",
  showlegend=False,
  x=[i for x in X_lowres for i in (x, x, None)],
  y=[i for y in Y_lowres for i in (0, y, None)],
  mode='lines',
  line=go.scatter.Line(color="grey")
  ))

fig.add_trace(go.Scatter(name= "Discrete Wave",
  # legendgroup="Discrete Wave",
  x=X_lowres,
  y=Y_lowres,
  mode='markers',
  marker=dict(
      size=5,
      color="black",
  )))


fig.show()
fig.write_image("01.png")
exit()

# X = np.linspace(0, n, n * 2)
# t = n // 3
# print(t)
# a = Y[t]
# k = 1
# P = np.linspace(0, 2 * np.pi, 25)
# F = n * 1 / t - P * n / (2 * np.pi * t)

# for i in range(P.shape[0]):
#   S = a * np.cos(2 * np.pi * F[i] * X / n + P[i])
#   plt.plot(X, S, "b-", linewidth=.5)

# t = 51
# print(t)
# a = Y[t]
# F = n * 1 / t - P * n / (2 * np.pi * t)
# for i in range(P.shape[0]):
#   S = a * np.cos(2 * np.pi * F[i] * X / n + P[i])
#   plt.plot(X, S, "r-", linewidth=.5)

# plt.show()
