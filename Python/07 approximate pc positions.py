import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import numpy.polynomial.polynomial as poly
from collections import Counter
from math import gcd
import hp as hp


'''==============='''
''' Read wav file '''
'''==============='''


name = "brass"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)

XX = np.arange(Xpos.size)
A = poly.polyfit(XX, Xpos, 1)
b, a = A

print("a=", a)

x0 = int(np.round(- b / a))
x1 = int(np.round((Xpos[-1]- b) / a))
XX = np.arange(x0, x1)

Xposlms = poly.polyval(XX, [b, a])
# stdlms = hp.std(Xposlms, Xpos)

# Xposfit, Yposfit = hp.linearize_pc(Xpos)
# stdfit = hp.std(Yposfit, Xpos)
# print(stdlms, stdfit)
FT = np.fft.rfft(W)
f = np.argmax(np.abs(FT))
print(n/a, f)

# exit()


'''============================================================================'''
'''                              PLOT LINES                                    '''
'''============================================================================'''


fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="Length",
  yaxis_title="Number of Ocurrences",

  # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
  # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=12
  )
)

fig.add_trace(
  go.Scatter(
    name="Xpos", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xpos,
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

# fig.add_trace(
#   go.Scatter(
#     name="Xposfit", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=Xposfit,
#     y=Yposfit,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="gray",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

fig.add_trace(
  go.Scatter(
    name="Xposlms", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=XX,
    y=Xposlms,
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


# fig.add_trace(
#   go.Scatter(
#     name="Xposlms + std", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=XX,
#     y=Xposlms + 3 * stdlms,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="blue",
#         dash="dash"
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="Xposlms - std", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=XX,
#     y=Xposlms - 3 * stdlms,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="blue",
#         dash="dash"
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="Xneg", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=Xpos,
#     y=Xneg,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="red",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

fig.show(config=dict({'scrollZoom': True}))

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="Length",
  yaxis_title="Number of Ocurrences",

  # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
  # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=12
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
    name="Positive Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xpos,
    y=W[Xpos],
    # fill="toself",
    mode="markers",
    line=dict(
        # width=1,
        color="red",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

XX = []
XXd = []
YY = []
for x in Xposlms:
  XX.append(x), XX.append(x), XX.append(None)
  XXd.append(a / 2 + x), XXd.append(a / 2 + x), XXd.append(None)
  YY.append(-amp), YY.append(amp), YY.append(None)

fig.add_trace(
  go.Scatter(
    name="Positive Frontier lms", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=XX,
    y=YY,
    # fill="toself",
    mode="lines",
    line=dict(
        # width=1,
        color="blue",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="  Positive Frontier lms", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=XXd,
    y=YY,
    # fill="toself",
    mode="lines",
    line=dict(
        # width=1,
        color="gray",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))