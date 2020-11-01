import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import numpy.polynomial.polynomial as poly
from collections import Counter
from math import gcd


'''==============='''
''' Read wav file '''
'''==============='''


name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)

Xpos, Xneg = Xpos.astype(np.int), Xneg.astype(np.int)

XX = np.arange(Xpos.size)
# A = poly.polyfit(XX, Xpos, 4)
# Xposfit = poly.polyval(XX, A)

AB = []
for i in range(Xpos.size):
  for j in range(i+1, Xpos.size):
    anum = Xpos[j] - Xpos[i]
    aden = j - i
    agcd = gcd(anum, aden)
    bnum = i * Xpos[j] - j * Xpos[i]
    bden = i - j
    bgcd = gcd(bnum, bden)
    AB.append((anum / agcd, aden / agcd, bnum / bgcd, bden / bgcd))


countsAB = Counter([(ab[0], ab[1]) for ab in AB])

# countsAB = Counter(AB)

afrac = countsAB.most_common(1)[0][0]

B = []
for ab in AB:
  if afrac == (ab[0], ab[1]):
    B.append(ab[2] / ab[3])

print(f"a num, a den = {countsAB.most_common(10)[0][0]}, total={len(AB)}")

a = afrac[0] / afrac[1]
b = np.average(np.array(B))

# posL = []
# for i in range(1, Xpos.size):
#   posL.append(Xpos[i] - Xpos[i - 1])

# negL = []
# for i in range(1, Xneg.size):
#   negL.append(Xneg[i] - Xneg[i - 1])

# np.histogram(Xneg, np.arange(Xneg.size))


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
    name="Xpos", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xpos,
    # fill="toself",
    mode="markers",
    marker=dict(
        # width=1,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Xposfit", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=XX,
    y=a * XX + b,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Xneg", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xneg,
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