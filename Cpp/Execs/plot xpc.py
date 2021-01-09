from os import linesep
import wave
import numpy as np
import plotly.graph_objects as go


name = "piano33"

Xpc = np.genfromtxt(name + "_Xpcs.csv", delimiter=",").astype(int)
L = Xpc[1:] - Xpc[:-1]

Xpc_2 = np.genfromtxt(name + "_Xpcs_best.csv", delimiter=",").astype(int)
L_2 = Xpc_2[1:] - Xpc_2[:-1]


FT = np.fft.rfft(L_2)
absFT = np.abs(FT)
avg = np.average(absFT)
for i in range(0, absFT.size):
  if np.abs(FT[i]) <= avg:
    FT[i] = 0 + 0 * 1j

L_3 = np.fft.irfft(FT)

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
    name="Xpc", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=XX,
    y=Xpc,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Xpc_2", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=XX,
    y=Xpc_2,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

Xpc_3 = [Xpc_2[0]]

for l in L_3:
  Xpc_3.append(Xpc_3[-1] + l)

fig.add_trace(
  go.Scatter(
    name="Xpc_3", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=XX,
    y=Xpc_3,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="blue",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))
