import numpy as np
import plotly.graph_objects as go

Xp = [0, 3, 6, 7, 9]
Yp = [3, 2, 4, 1, 5]

def bezier(Xp, Yp):
  X = []
  Y = []
  for i in range(len(Xp) - 2):
    print(i)
    '''y = m x + b'''
    m0 = (Yp[i + 1] - Yp[i]) / (Xp[i + 1] - Xp[i])
    b0 = Yp[i] - m0 * Xp[i]
    f0 = lambda x : m0 * x + b0

    m1 = (Yp[i + 2] - Yp[i + 1]) / (Xp[i + 2] - Xp[i + 1])
    b1 = Yp[i + 1] - m1 * Xp[i + 1]
    f1 = lambda x : m1 * x + b1

    xx = np.arange(Xp[i], Xp[i + 1] + 1, 1)
    for x in xx:
      X.append(x)
      Y.append(f0(x))
    X.append(None)
    Y.append(None)

    xx = np.arange(Xp[i + 1], Xp[i + 2] + 1, 1)
    for x in xx:
      X.append(x)
      Y.append(f1(x))
    X.append(None)
    Y.append(None)
  print(X)
  return X, Y

X, Y = bezier(Xp, Yp)

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''

fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",
  # yaxis = dict(
  #   scaleanchor = "x",
  #   scaleratio = 1,
  # ),
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

fig.add_trace(
  go.Scatter(
    name="Control Points",
    x=Xp,
    y=Yp,
    mode="markers",
    marker=dict(
        size=8,
        color="red",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="Bezier",
    x=X,
    y=Y,
    mode="markers+lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))