import numpy as np
import plotly.graph_objects as go




before = np.genfromtxt("00before.csv", delimiter=",")

after = np.genfromtxt("01after.csv", delimiter=",")

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

  # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
  # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

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
    name="before", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=np.arange(W.size),
    y=before,
    mode="lines+markers",
    line=dict(
        # size=8,
        color="blue",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="before shift", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=np.arange(before.size) + before.size,
    y=before,
    mode="lines+markers",
    line=dict(
        # size=8,
        color="blue",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="after", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=np.arange(W.size),
    y=after,
    mode="lines+markers",
    line=dict(
        # size=8,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="after shift", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=np.arange(after.size) + after.size,
    y=after,
    mode="lines+markers",
    line=dict(
        # size=8,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))
