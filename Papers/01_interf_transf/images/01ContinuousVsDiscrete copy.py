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
fig.write_image("./01ContinuousVsDiscrete.png")
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