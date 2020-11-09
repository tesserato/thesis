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

#   ftpos = np.zeros(nft, dtype=np.complex)
#   for i in range(1, Xpos.size):
#     x0 = Xpos[i - 1]
#     x1 = Xpos[i]
#     if x1 - x0 >= 4:
#       yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
#       Ylocal = yx(Xlocal)
#       ftpos += np.fft.rfft(Ylocal)

#   ftneg = np.zeros(nft, dtype=np.complex)
#   for i in range(1, Xneg.size):
#     x0 = Xneg[i - 1]
#     x1 = Xneg[i]
#     if x1 - x0 >= 4:
#       yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
#       Ylocal = yx(Xlocal)
#       ftneg += np.fft.rfft(Ylocal)

#   ####### SHIFT #######
#   f = np.argmax(np.abs(ftneg))
#   p = np.angle(ftneg[f])
#   print(f"f={f}, p={p}")

#   Y = np.fft.irfft(ftneg, maxT)
#   d = maxT - np.argmax(Y)
#   x = 2 * np.pi * np.arange(nft) / maxT
#   ftneg = np.exp(-1j * x * d) * ftneg
#   #####################
#   Y = np.fft.irfft(ftpos + ftneg, maxT)
#   Y = Y / np.max(np.abs(Y))
#   return Y

'''==============='''
''' Read wav file '''
'''==============='''

name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

'''==============='''
'''     Normal    '''
'''==============='''

Xpos, Xneg = se.get_frontiers(W)

Xpos = hp.refine_frontier_iter(Xpos, W)
Xneg = hp.refine_frontier_iter(Xneg, W)

print("Sizes:", Xpos.size, Xneg.size)

Y, _, _ = hp.average_pc_waveform(Xpos, Xneg, W)


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

fig.add_trace(
  go.Scattergl(
    name="+Pseudo-Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pospseudoCyclesY for item in Xpc + [None]],
    y=Y,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))