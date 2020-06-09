import plotly.graph_objects as go
import numpy as np


#######################
n = 30

X = np.arange(n)

f = 3
p = np.pi / 2
W = np.cos(p + 2 * np.pi * f * X / n)
#######################

fig = go.Figure()

# fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
# fig.update_yaxes(tickvals=[i for i in range(n)])

# fig.update_layout(
#     # width = 2 * np.pi * 220,
#     # height = n * 220,
#     # yaxis = dict(scaleanchor = "x", scaleratio = 1),
#     title=f"n = {n}",
#     xaxis_title="Phase",
#     yaxis_title="Frequency",
#     font=dict(
#         family="Courier New, monospace",
#         size=18,
#         color="#7f7f7f"
#     )
# )

X = []
Y = []
Z = []
R = []

t=29
for f in np.linspace(0, n / 2, 200):
  for p in np.linspace(0, 2 * np.pi, 200):
    r = np.abs(np.cos(p + 2 * np.pi * f * t / n))
    x = np.cos(p) * r
    y = np.sin(p) * r
    z = f

    X.append(x)
    Y.append(y)
    Z.append(z)
    R.append(r)

fig.add_trace(
  go.Scatter3d(
    x=X, y=Y, z=Z,
    mode='markers',
    marker=dict(
        size=2,
        color=R,
        colorscale='Viridis'
    )
  )
)

fig.show()



