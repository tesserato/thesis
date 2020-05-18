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
    # title="Continuous x Discrete Wave",
    xaxis_title="Time",
    yaxis_title="Amplitude")

fig.update_yaxes(tickvals=np.linspace(-1, 1, 21), zerolinewidth=2, zerolinecolor='black')

fig.update_layout(legend_orientation="h")

fig.add_trace(go.Scatter(name= "Continuous Wave",
  x=X,
  y=Y,
  mode='lines',
  line_shape='spline',
  line=go.scatter.Line(color="gray", width=2)
  ))

fig.add_trace(go.Scatter(name= "Discrete Wave Stem",
  showlegend=False,
  x=[i for x in X_lowres for i in (x, x, None)],
  y=[i for y in Y_lowres for i in (0, y, None)],
  mode='lines',
  line=go.scatter.Line(color="black", width=1)
  ))

fig.add_trace(go.Scatter(name= "Discrete Wave",
  x=X_lowres,
  y=Y_lowres,
  mode='markers',
  marker=dict(
      size=5,
      color="black",
  )))

# fig.show()
fig.write_image("./01ContinuousVsDiscrete.eps", width=800, height=400, scale=10)