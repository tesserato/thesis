import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import numpy.polynomial.polynomial as poly
from collections import Counter
from math import gcd
import hp as hp
from scipy.signal import savgol_filter
from plotly.subplots import make_subplots


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
Xpossmooth = savgol_filter(Xpos, 5, 3)

L = Xpos[1:] - Xpos[:-1]
avgL = np.average(L)
stdL = np.std(L)

Xnew = []
for i in range(1, Xpos.size):
  x0 = int(Xpos[i - 1])
  x1 = int(Xpos[i])
  if x1 - x0 > avgL + 2 * stdL:
    Xzeroes = []
    currsign = np.sign(W[x0])
    for i in range(x0 + 1, x1):
      if currsign != np.sign(W[i]):
        Xzeroes.append(i)
        currsign = np.sign(W[i])
    if len(Xzeroes) > 1:
      Xnew.append(Xzeroes[0] + np.argmax(W[Xzeroes[0] : Xzeroes[-1]]))
Xnew = np.array(Xnew, dtype=np.int)

# XX = np.arange(Xpos.size)
# A = poly.polyfit(XX, Xpos, 1)
# b, a = A

# print("a=", a)

# x0 = int(np.round(- b / a))
# x1 = int(np.round((Xpos[-1]- b) / a))
# XX = np.arange(x0, x1)

# Xposlms = poly.polyval(XX, [b, a])
# # stdlms = hp.std(Xposlms, Xpos)

# # Xposfit, Yposfit = hp.linearize_pc(Xpos)
# # stdfit = hp.std(Yposfit, Xpos)
# # print(stdlms, stdfit)
# FT = np.fft.rfft(W)
# f = np.argmax(np.abs(FT))
# print(n/a, f)

# exit()

posL = []
for i in range(1, Xpos.size):
  posL.append(Xpos[i] - Xpos[i - 1])


'''============================================================================'''
'''                              PLOT LINES                                    '''
'''============================================================================'''


fig = make_subplots(rows=2, cols=1, shared_yaxes=True)
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
  ),row=1,col=1
)


fig.add_trace(
  go.Scatter(
    name="Xposooth", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=XX,
    y=Xpossmooth,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="blue",
        # showscale=False
    ),
    # visible = "legendonly"
  ),row=1,col=1
)

fig.add_trace(
  go.Scatter(
    name="Xposooth", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=XX,
    y=posL,
    # fill="toself",
    mode="markers",
    line=dict(
        width=1,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  ),row=2,col=1
)

fig.show(config=dict({'scrollZoom': True}))

# exit()
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

fig.add_trace(
  go.Scatter(
    name="Positive Frontier New", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xnew,
    y=W[Xnew],
    # fill="toself",
    mode="markers",
    line=dict(
        # width=1,
        color="black",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

Xr = [0, 0, None]
Yr = [-amp, amp, None]
for i in range(1, Xpossmooth.size):
  x = (Xpossmooth[i - 1] + Xpossmooth[i]) / 2
  Xr.append(x), Xr.append(x), Xr.append(None)
  Yr.append(-amp), Yr.append(amp), Yr.append(None)

XX = []
YY = []
for x in Xpossmooth:
  XX.append(x), XX.append(x), XX.append(None)
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
    x=Xr,
    y=Yr,
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