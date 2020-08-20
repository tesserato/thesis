import numpy.polynomial.polynomial as poly
import sys
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
import numpy as np
from scipy import signal

import plotly.graph_objects as go

from Helper import signal_to_pulses, get_pulses_area

'''Generating Wave'''

np.random.seed(1)
# fps = 10
n = 1000+1

X = np.linspace(0, 1, n)
W = np.zeros(n)

afp = [
  [3, 10, np.pi],
  [2, 2, 1.2 * np.pi],
  [1, 2.3, 1.4 * np.pi]
  ]
for a, f, p in afp:
  W += a * np.cos(p + 2 * np.pi * f * X / n)# + np.random.normal(0, a / 10, n)

# coefs = poly.polyfit([0, n//4, 3 * n//4, n], [0.0, 0.1, 2.0, 2.0], 3)
# A = poly.polyval(X, coefs)
# W += A

W = signal.sawtooth(2 * np.pi * 5 * X)

'''Wave to pulses'''
pulses = signal_to_pulses(W)
pulses_X, pulses_Y = get_pulses_area(pulses)


'''Plotting'''
fig = go.Figure()

fig.layout.template ="plotly_white"
 
fig.update_layout(
  xaxis_title="$i$",
  yaxis_title="Amplitude",
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
    name="Samples",
    x=X,
    y=W,
    mode='lines',
    line=dict(
        width=5,
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

''' Pseudo-Cycles '''
cycles_X = []
cycles_Y = []
for x in pulses_X[2:-3:4]:
  cycles_X.append(x)   , cycles_Y.append(-1)
  cycles_X.append(x)   , cycles_Y.append(1)
  cycles_X.append(None), cycles_Y.append(None)

fig.add_trace(
  go.Scatter(
    name="Pseudo-Cycles",
    x=cycles_X,
    y=cycles_Y,
    # marker_symbol="line-ns",
    mode="lines",
    line=dict(
      # width=5,
      color="black",
      dash="dash"
    )
    # marker_line_width=2, 
    # marker_size=50,
  )
)

fig.show()
# fig.write_image("./03Frontiers.pdf", width=800, height=400, scale=1)
