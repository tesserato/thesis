import plotly.graph_objects as go
import numpy as np
import signal_envelope as se




'''==============='''
''' Read wav file '''
'''==============='''


name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)

posL = []
for i in range(1, Xpos.size):
  posL.append(Xpos[i] - Xpos[i - 1])

negL = []
for i in range(1, Xneg.size):
  negL.append(Xneg[i] - Xneg[i - 1])

L = np.array(posL + negL)
avgL = np.average(L)
stdL = np.std(L)




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
    name="+Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=posL,
    # fill="toself",
    mode="markers",
    marker=dict(
        size=3,
        color="black",
        showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="-Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=negL,
    # fill="toself",
    mode="markers",
    marker=dict(
        size=3,
        color="red",
        showscale=False
    ),
    # visible = "legendonly"
  )
)


fig.add_trace(
  go.Scatter(
    name="Average L", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, max(Xpos.size, Xneg.size)],
    y=[avgL, avgL],
    # fill="toself",
    mode="lines",
    # line=dict(
    #     # size=3,
    #     color="black",
    #     showscale=False
    # ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+std L", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, max(Xpos.size, Xneg.size)],
    y=[avgL + stdL, avgL + stdL],
    # fill="toself",
    mode="lines",
    # line=dict(
    #     # size=3,
    #     color="black",
    #     showscale=False
    # ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="-std L", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, max(Xpos.size, Xneg.size)],
    y=[avgL - stdL, avgL - stdL],
    # fill="toself",
    mode="lines",
    # line=dict(
    #     # size=3,
    #     color="black",
    #     showscale=False
    # ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))