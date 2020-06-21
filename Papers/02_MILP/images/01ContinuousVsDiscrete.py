import numpy as np
import random
import plotly.graph_objects as go


random.seed(0)

fps = 44100
n = 1000+1

X = np.arange(n) / fps
Y = np.zeros(n)

for _ in range(100):
  a = random.uniform(1, 5)
  f = random.uniform(1, 10)
  p = random.uniform(0, 2 * np.pi)
  Y += a * np.cos(p + 2 * np.pi * f * X * fps / n)

Y = Y / np.max(np.abs(Y))

X_lowres = X[::20]
Y_lowres = np.round(Y[::20], 1)

fig = go.Figure()

fig.layout.template ="plotly_white"
 
fig.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Amplitude",
    legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
    margin=dict(l=5, r=5, b=5, t=5),
    font=dict(
    family="Computer Modern",
    color="black",
    size=18
    ))

fig.update_xaxes(
  gridwidth=1, 
  gridcolor='silver'
)

fig.update_yaxes(tickvals=np.linspace(-1, 1, 11), zerolinewidth=2, zerolinecolor='gray')

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

fig.show()
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)