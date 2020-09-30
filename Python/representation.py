import plotly.graph_objects as go
import numpy as np
import signal_envelope as se




'''==============='''
''' Read wav file '''
'''==============='''


name = "brass"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)

pseudoCyclesX = []
pseudoCyclesY = []

posL = []
for i in range(1, Xpos.size):
  x0 = Xpos[i - 1]
  x1 = Xpos[i]
  posL.append(x1 - x0)
  a = np.max(np.abs(W[x0 : x1]))
  for j in range(x1 - x0):
    pseudoCyclesX.append(j)
    pseudoCyclesY.append(W[int(x0+j)] / a)
  pseudoCyclesX.append(None)
  pseudoCyclesY.append(None)

negL = []
for i in range(1, Xneg.size):
  x0 = Xneg[i - 1]
  x1 = Xneg[i]
  negL.append(x1 - x0)
  a = np.max(np.abs(W[x0 : x1]))
  for j in range(x1 - x0):
    pseudoCyclesX.append(j)
    pseudoCyclesY.append(-W[int(x0+j)] / a)
  pseudoCyclesX.append(None)
  pseudoCyclesY.append(None)

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
    x=pseudoCyclesX,
    y=pseudoCyclesY,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="rgba(38, 12, 12, 0.2)",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))


exit()
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
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xpos,
    y=W[Xpos],
    # fill="toself",
    mode="markers",
    line=dict(
        # width=1,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="-Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xneg,
    y=W[Xneg],
    # fill="toself",
    mode="markers",
    line=dict(
        # width=1,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))