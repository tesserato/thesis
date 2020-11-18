from numpy.lib.function_base import append
import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp
from statistics import mode
from scipy import interpolate

# def average_pc_waveform(Xpos, Xneg, W):
#   T = []
#   for i in range(1, Xpos.size):
#     T.append(Xpos[i] - Xpos[i - 1])
#   for i in range(1, Xneg.size):
#     T.append(Xneg[i] - Xneg[i - 1])
#   T = np.array(T, dtype = np.int)
#   maxT = np.max(T)

#   nft = int(maxT//2 + 1)
#   Xlocal = np.linspace(0, 1, maxT)

#   pos_orig_pcs = []
#   pos_norm_pcs = []
#   pos_fts = []
#   ftpos = np.zeros(nft, dtype=np.complex)
#   for i in range(1, Xpos.size):
#     x0 = Xpos[i - 1]
#     x1 = Xpos[i]
#     pos_orig_pcs.append(W[x0 : x1])
#     if x1 - x0 >= 4:
#       yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
#       Ylocal = yx(Xlocal)
#       ft = np.fft.rfft(Ylocal) / W[x0 : x1].size * 2
#       ftpos += ft
#       pos_norm_pcs.append(Ylocal)
#       pos_fts.append(ft)

#   neg_orig_pcs = []
#   neg_norm_pcs = []
#   neg_fts = []
#   ftneg = np.zeros(nft, dtype=np.complex)
#   for i in range(1, Xneg.size):
#     x0 = Xneg[i - 1]
#     x1 = Xneg[i]
#     neg_orig_pcs.append(W[x0 : x1])
#     if x1 - x0 >= 4:
#       yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
#       Ylocal = yx(Xlocal)
#       ft = np.fft.rfft(Ylocal)
#       ftneg += ft
#       neg_norm_pcs.append(Ylocal)
#       neg_fts.append(ft)

#   ####### SHIFT #######
#   f = np.argmax(np.abs(ftneg))
#   p = np.angle(ftneg[f])
#   print(f"f={f}, p={p}")

#   avgpc = np.fft.irfft(ftneg, maxT)
#   d = maxT - np.argmax(avgpc)
#   x = 2 * np.pi * np.arange(nft) / maxT
#   ftneg = np.exp(-1j * x * d) * ftneg
#   ######################
#   avgpc = np.fft.irfft(ftpos)
#   # avgpc = avgpc / np.max(np.abs(avgpc))
#   return avgpc, pos_orig_pcs, pos_norm_pcs, pos_fts, neg_orig_pcs, neg_norm_pcs, neg_fts

def average_pc_waveform(Xpos, Xneg, W):
  T = []
  for i in range(1, Xpos.size):
    T.append(Xpos[i] - Xpos[i - 1])
  for i in range(1, Xneg.size):
    T.append(Xneg[i] - Xneg[i - 1])
  T = np.array(T, dtype = np.int)
  maxT = np.max(T)

  Xlocal = np.linspace(0, 1, maxT)

  pos_orig_pcs = []
  pos_norm_pcs = []
  for i in range(1, Xpos.size):
    x0 = Xpos[i - 1]
    x1 = Xpos[i]
    pos_orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      pos_norm_pcs.append(Ylocal)

  neg_orig_pcs = []
  neg_norm_pcs = []
  for i in range(1, Xneg.size):
    x0 = Xneg[i - 1]
    x1 = Xneg[i]
    neg_orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      neg_norm_pcs.append(Ylocal)
  return pos_orig_pcs, pos_norm_pcs, neg_orig_pcs, neg_norm_pcs



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

transp_black = "rgba(38, 12, 12, 0.2)"

'''==============='''
''' Read wav file '''
'''==============='''

name = "tom"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)
Xpos = hp.refine_frontier_iter(Xpos, W)
Xneg = hp.refine_frontier_iter(Xneg, W)

print("Sizes:", Xpos.size, Xneg.size)

pos_orig_pcs, pos_norm_pcs, neg_orig_pcs, neg_norm_pcs = average_pc_waveform(Xpos, Xneg, W)

pos_avgpc = np.average(np.array(pos_norm_pcs), 0)
neg_avgpc = np.average(np.array(neg_norm_pcs), 0)

# avgpc = avgpc / 7

wave = []
a = np.max(np.abs(pos_avgpc))

for i in range(Xpos.size):
  amp = np.max(np.abs(pos_avgpc))
  w = a * pos_avgpc / amp
  a = w[-1]
  for j in w:
    wave.append(j)

wave = np.array(wave)

se.save_wav(wave)

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
  # yaxis = dict(scaleanchor = "x", scaleratio = 1 ),                   # <|<|<|<|<|<|<|<|<|<|<|<|
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

X, Y = to_plot(pos_orig_pcs)
fig.add_trace(
  go.Scattergl(
    name="Original Positive Pseudo Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color=transp_black,
        # showscale=False
    ),
    visible = "legendonly"
  )
)

X, Y = to_plot(pos_norm_pcs)
fig.add_trace(
  go.Scattergl(
    name="Normalized Positive Pseudo Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color=transp_black,
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

X, Y = to_plot(neg_norm_pcs)
fig.add_trace(
  go.Scattergl(
    name="Normalized Negative Pseudo Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color=transp_black,
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="Average Positive Waveform", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pospseudoCyclesY for item in Xpc + [None]],
    y=pos_avgpc,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="Average Negative Waveform", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pospseudoCyclesY for item in Xpc + [None]],
    y=neg_avgpc,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="Average Waveform 2", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=np.arange(pos_avgpc.size) + pos_avgpc.size - 1,
    y=pos_avgpc,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))