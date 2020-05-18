import numpy as np
import random
import plotly.graph_objects as go


random.seed(0)

fps = 44100
n = 35+1

X = np.arange(n)
W = np.zeros(n)

for _ in range(100):
  a = random.uniform(1, 5)
  f = random.uniform(1, 10)
  p = random.uniform(0, 2 * np.pi)
  W += a * np.cos(p + 2 * np.pi * f * X / n)

W = W / np.max(np.abs(W))



fig = go.Figure()

fig.layout.template ="plotly_white"
 
fig.update_layout(
    xaxis_title="Frame",
    yaxis_title="Amplitude",
    legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
    margin=dict(l=5, r=5, b=5, t=5))


fig.update_yaxes(zerolinewidth=2, zerolinecolor='black')


t = 3
a = W[t]
P = np.linspace(0, 2 * np.pi, 25)
F = n * 1 / t - P * n / (2 * np.pi * t)

for i in range(P.shape[0]):
  T = np.linspace(0, 12.5, 100)
  S = a * np.cos(2 * np.pi * F[i] * T / n + P[i])

  fig.add_trace(go.Scatter(name=f"t={t}, f={F[i]}, p={P[i]}",
  showlegend=False,
  x=T,
  y=S,
  mode='lines',
  line_shape='spline',
  line=go.scatter.Line(color="gray", width=1)
  ))



t = 28
a = W[t]
P = np.linspace(0, 2 * np.pi, 25)
F = n * 1 / t - P * n / (2 * np.pi * t)

for i in range(P.shape[0]):
  T = np.linspace(18.5, n, 100)
  S = a * np.cos(2 * np.pi * F[i] * T / n + P[i])

  fig.add_trace(go.Scatter(name=f"t={t}, f={F[i]}, p={P[i]}",
  showlegend=False,
  x=T,
  y=S,
  mode='lines',
  line_shape='spline',
  line=go.scatter.Line(color="gray", width=1)
  ))



fig.add_trace(go.Scatter(name= "Discrete Wave Stem",
  showlegend=False,
  x=[i for x in X for i in (x, x, None)],
  y=[i for y in W for i in (0, y, None)],
  mode='lines',
  line=go.scatter.Line(color="black", width=1)
  ))

fig.add_trace(go.Scatter(name= "Discrete Wave",
  x=X,
  y=W,
  mode='markers',
  marker=dict(
      size=5,
      color="black",
  )))

# fig.show()
fig.write_image("./03OptSinusoids.png", width=800, height=400, scale=10)
fig.write_image("./03OptSinusoids.eps", width=800, height=400, scale=10)