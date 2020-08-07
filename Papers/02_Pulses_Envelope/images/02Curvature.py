# import random
# random.seed(2)
import sys
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
# import site
# site.addsitedir("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
import numpy as np

import plotly.graph_objects as go

from Helper import signal_to_pulses, get_pulses_area, split_pulses

def get_parabola_coefficients(X, Y):
  assert len(X) == len(Y) and len(X) == 3
  x0, x1, x2 = X[0], X[1], X[2]
  y0, y1, y2 = Y[0], Y[1], Y[2]

  den = (-x0 + x2) * (x0 - x1) * (x1 - x2)

  a = ((x0 - x1) * (y1 - y2) - (x1 - x2) * (y0 - y1)) / (2 * den)
  b = (-x0 * (x0 - x1) * (y1 - y2) + x2 * (x1 - x2) * (y0 - y1)) / den
  c = -a * x0**2 - b * x0 + y0
  return a, b, c

def get_approximation(X, Y):
  Y = np.abs(Y)
  final_X = []
  final_Y = []
  for i in range(1, len(X) - 1):
    x0, x1, x2 = (X[i - 1] + X[i]) / 2, X[i], (X[i] + X[i + 1]) / 2
    y0, y1, y2 = (Y[i - 1] + Y[i]) / 2, Y[i], (Y[i] + Y[i + 1]) / 2
    a, b, c = get_parabola_coefficients([x0, x1, x2], [y0, y1, y2])
    XX = np.linspace(x0, x2, 10)
    YY = a * XX**2 + b * XX + c
    for x, y in zip(XX, YY):
      final_X.append(x), final_Y.append(y)
    final_X.append(None), final_Y.append(None)
  return final_X, final_Y

'''Generating Wave'''
np.random.seed(1)
# fps = 10
n = 200+1

X = np.arange(n)# / fps
W = np.zeros(n)

afp = [
  [1, 8, np.pi],
  # [1, 2, 1.2 * np.pi],
  # [1, 2.3, 1.4 * np.pi]
  ]
for a, f, p in afp:
  W += a * np.cos(p + 2 * np.pi * f * X / n) + np.random.normal(0, a / 5, n)

W = W / np.max(np.abs(W))

'''Wave to pulses'''
pulses = signal_to_pulses(W)
pulses_X, pulses_Y = get_pulses_area(pulses)


pos_pulses, neg_pulses, pos_noises, neg_noises = split_pulses(pulses)
pos_X, neg_X = np.array([p.x for p in pos_pulses]), np.array([p.x for p in neg_pulses])
pos_Y, neg_Y = np.array([p.y for p in pos_pulses]), np.array([p.y for p in neg_pulses])
# pos_L, neg_L = np.array([p.len for p in pos_pulses]) , np.array([p.len for p in neg_pulses])


scaling = np.average([p.len for p in pulses]) / np.average([abs(p.y) for p in pulses])
pos_Y = pos_Y * scaling

neg_Y = neg_Y * scaling

W = W * scaling
pulses_Y = np.array(pulses_Y) * scaling

XX, YY = get_approximation(pos_X, pos_Y)

print(XX)

'''Plotting'''
fig = go.Figure()

fig.layout.template ="plotly_white"
 
fig.update_layout(
  xaxis_title="$i$",
  yaxis_title="Normalized Amplitude",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=2, zerolinecolor='gray')

'''Samples'''
fig.add_trace(
  go.Scatter(
    name= "Samples Stem",
    showlegend=False,
    x=[i for x in X for i in (x, x, None)],
    y=[i for y in W for i in (0, y, None)],
    mode='lines',
    line=dict(
        width=1,
        color="black",
    )
  )
)
fig.add_trace(
  go.Scatter(
    name="Samples",
    x=X,
    y=W,
    mode='markers',
    marker=dict(
        size=5,
        color="black",
    )
  )
)

''' Pulses '''
fig.add_trace(
  go.Scatter(
    name="Pulses",
    x=pulses_X,
    y=pulses_Y,
    fill="tozeroy",
    mode="none",
    fillcolor="rgba(0,0,0,0.16)",
    # visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Samples",
#     x=pos_X,
#     y=pos_Y,
#     mode='markers',
#     marker=dict(
#         size=20,
#         color="black",
#     )
#   )
# )

'''Approx'''
fig.add_trace(
  go.Scatter(
    name= "Approx",
    x=XX,
    y=YY,
    mode='lines',
    line=dict(
        width=1,
        color="black",
    )
  )
)

fig.show()
# fig.write_image("./02Curvature.pdf", width=800, height=400, scale=1)
