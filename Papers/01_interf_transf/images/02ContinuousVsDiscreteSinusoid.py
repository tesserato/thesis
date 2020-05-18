import numpy as np
import random
import plotly.graph_objects as go

a = 1
F = 2
p = 1
fps=31

''' CONTINUOUS '''
t_f = 2 # seconds
T = np.linspace(0, t_f, 1000)
S = a * np.cos(p + 2 * np.pi * F * T)

''' DISCRETE '''
n = fps * t_f
X = np.arange(n)
f = n * F / fps
W = a * np.cos(p + 2 * np.pi * f * X / n)

fig = go.Figure()

fig.layout.template ="plotly_white"

fig.update_layout(
  xaxis_title="Frame number, Time (seconds)",
  yaxis_title="Amplitude",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(family="Computer Modern", color="black", size=18)
  )

stp = 5

fig.update_xaxes(
  gridwidth=1, 
  gridcolor='silver',
  ticktext=[f"i={i}, t={round(i/fps, 3)}" for i in X[::stp]],
  tickvals=X[::stp] / fps
)

fig.update_yaxes(zerolinewidth=2, zerolinecolor='silver', automargin=True)

fig.add_trace(go.Scatter(name=f"Continuous Wave",
  x=T,
  y=S,
  mode='lines',
  line_shape='spline',
  line=go.scatter.Line(color="gray", width=12)
  ))

fig.add_trace(go.Scatter(name=f"Discrete Wave",
  x=X/fps,
  y=W,
  mode='lines+markers',
  # line_shape='spline',
  line=go.scatter.Line(color="black", width=1)
  ))

# fig.show()
fig.write_image("./02ContinuousVsDiscreteSinusoid.pdf", width=800, height=400, scale=1)