from statistics import mode
import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import numpy.polynomial.polynomial as poly
# import numpy.polynomial.polynomial as poly
# from collections import Counter
# from math import gcd
import hp as hp
from scipy.signal import savgol_filter
# from plotly.subplots import make_subplots
from scipy import interpolate
from statistics import mode

def to_plot(Matrix):
  X = []
  Y = []
  for line in Matrix:
    for x, y in enumerate(line):
      X.append(x)
      Y.append(y)
    X.append(None)
    Y.append(None)
  return X, Y

def average_pc_waveform(Xp, W):
  maxT = np.max(np.abs(Xp[1:] - Xp[:-1]))

  Xlocal = np.linspace(0, 1, maxT)

  pos_orig_pcs = []
  pos_norm_pcs = []
  for i in range(1, Xp.size):
    x0 = Xp[i - 1]
    x1 = Xp[i] + 1
    pos_orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
      # Ylocal = yx(Xlocal)
      pos_norm_pcs.append(yx(Xlocal))

  pos_avgpc = np.average(np.array(pos_norm_pcs), 0)
  return pos_avgpc, pos_orig_pcs, pos_norm_pcs

def find_zeroes(V):
  s = np.sign(V[0])
  zeroes = []
  for i, v in enumerate(V):
    if np.sign(v) != s or v == 0:
      s = np.sign(-1 * s)
      zeroes.append(i)
  return np.array(zeroes, dtype=np.int)

'''==============='''
''' Read wav file '''
'''==============='''

name = "alto"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos_orig, Xneg_orig = se.get_frontiers(W)
Xpos = hp.refine_frontier_iter(Xpos_orig, W)
Xneg = hp.refine_frontier_iter(Xneg_orig, W)

nn = min(Xpos.size, Xneg.size)
Xpc = ((Xpos[:nn] + Xneg[:nn]) / 2).astype(np.float64)

XX = np.arange(nn, dtype=np.uint64)
b, a = poly.polyfit(XX, Xpc, 1)
Xpc_linear = a * XX + b

Xdev = Xpc_linear[:nn] - Xpc[:nn]
av = np.average(Xdev)
Xdev = Xdev - av

zeroes = find_zeroes(Xdev)

A = hp.constrained_least_squares_arbitrary_intervals(XX, Xdev, zeroes.tolist(), 3)
Xdev_est = hp.coefs_to_array_arbitrary_intervals(A, XX, zeroes.tolist(), nn) + av

print("e2", np.average(np.abs(Xdev)))

Xpcs = np.round(Xpc_linear[:nn] + Xdev[:nn] + av)#.astype(np.int)
Xdev = Xdev + av
'''============================================================================'''
'''                              PLOT LINES                                    '''
'''============================================================================'''

fig = fig = go.Figure()
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

# transp_black = "rgba(38, 12, 12, 0.2)"
fig.add_trace(
  go.Scattergl(
    name="X deviation from straight line", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=np.arange(Xdev.size),
    y=Xdev,
    # fill="toself",
    mode="lines+markers",
    line=dict(
        width=1,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="X deviation reconstructed", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=np.arange(Xdev.size),
    y=Xdev_est,
    # fill="toself",
    mode="lines+markers",
    line=dict(
        width=1,
        color="blue",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

X = []
Y = []
ma = np.max(Xdev)
mi = np.min(Xdev)
for x in zeroes:
  X.append(x)
  X.append(x)
  X.append(None)
  Y.append(mi)
  Y.append(ma)
  Y.append(None)

fig.add_trace(
  go.Scatter(
    name="divs", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
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
fig.show(config=dict({'scrollZoom': True}))