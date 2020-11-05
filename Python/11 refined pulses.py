import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp

'''==============='''
''' Read wav file '''
'''==============='''

name = "tom"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

'''==============='''
'''     Normal    '''
'''==============='''

Xpos, Xneg = se.get_frontiers(W)

maxL = []
for i in range(1, Xpos.size):
  maxL.append(Xpos[i] - Xpos[i - 1])
for i in range(1, Xneg.size):
  maxL.append(Xneg[i] - Xneg[i - 1])
maxL = np.max(np.array(maxL))
print(f"Max L = {maxL}")

Xpc = np.arange(maxL)

pospseudoCyclesX = []
pospseudoCyclesY = []
for i in range(1, Xpos.size):
  x0 = Xpos[i - 1]
  x1 = Xpos[i]
  # a = np.max(np.abs(W[x0 : x1]))
  ft = np.fft.rfft(W[x0 : x1])
  npulse = np.fft.irfft(ft, maxL)
  pospseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
  pospseudoCyclesX.append(Xpc)
pospseudoCyclesX = np.array(pospseudoCyclesX)
pospseudoCyclesY = np.array(pospseudoCyclesY)

negpseudoCyclesX = []
negpseudoCyclesY = []
for i in range(1, Xneg.size):
  x0 = Xneg[i - 1]
  x1 = Xneg[i]
  # a = np.max(np.abs(W[x0 : x1]))
  ft = np.fft.rfft(W[x0 : x1])
  npulse = np.fft.irfft(ft, maxL)
  negpseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
  negpseudoCyclesX.append(Xpc)
negpseudoCyclesX = np.array(negpseudoCyclesX)
negpseudoCyclesY = np.array(negpseudoCyclesY)


'''==============='''
'''    Refined    '''
'''==============='''

Xposadicional = hp.refine_frontier(Xpos, W)
Xpos = np.sort(np.hstack([Xpos, Xposadicional])).astype(np.int)

Xnegadicional = hp.refine_frontier(Xneg, W)
Xneg = np.sort(np.hstack([Xneg, Xnegadicional])).astype(np.int)

print("sizes:",Xpos.size, Xneg.size)

# maxL = []
# for i in range(1, Xpos.size):
#   maxL.append(Xpos[i] - Xpos[i - 1])
# for i in range(1, Xneg.size):
#   maxL.append(Xneg[i] - Xneg[i - 1])
# maxL = np.max(np.array(maxL))
# print(f"Max L = {maxL}")

refpospseudoCyclesX = []
refpospseudoCyclesY = []
for i in range(1, Xpos.size):
  x0 = Xpos[i - 1]
  x1 = Xpos[i]
  # a = np.max(np.abs(W[x0 : x1]))
  ft = np.fft.rfft(W[x0 : x1])
  npulse = np.fft.irfft(ft, maxL)
  refpospseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
  refpospseudoCyclesX.append(Xpc)
refpospseudoCyclesX = np.array(refpospseudoCyclesX)
refpospseudoCyclesY = np.array(refpospseudoCyclesY)

refnegpseudoCyclesX = []
refnegpseudoCyclesY = []
for i in range(1, Xneg.size):
  x0 = Xneg[i - 1]
  x1 = Xneg[i]
  # a = np.max(np.abs(W[x0 : x1]))
  ft = np.fft.rfft(W[x0 : x1])
  npulse = np.fft.irfft(ft, maxL)
  refnegpseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
  refnegpseudoCyclesX.append(Xpc)
refnegpseudoCyclesX = np.array(refnegpseudoCyclesX)
refnegpseudoCyclesY = np.array(refnegpseudoCyclesY)


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
    x=[item for sublist in pospseudoCyclesX for item in sublist.tolist() + [None]],
    y=[item for sublist in pospseudoCyclesY for item in sublist.tolist() + [None]],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="rgba(38, 12, 12, 0.2)",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="-Pseudo-Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[item for sublist in negpseudoCyclesX for item in sublist.tolist() + [None]],
    y=[item for sublist in negpseudoCyclesY for item in sublist.tolist() + [None]],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="rgba(38, 12, 12, 0.2)",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="+PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=np.average(pospseudoCyclesY, 0),
    # fill="toself",
    mode="lines",
    line=dict(
        width=4,
        color="red",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

negpseudoCyclesY_avg = np.average(negpseudoCyclesY, 0)
idx = np.argmax(negpseudoCyclesY_avg)
negpseudoCyclesY_avg = np.roll(negpseudoCyclesY_avg, -idx)

fig.add_trace(
  go.Scattergl(
    name="-PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=negpseudoCyclesY_avg,
    # fill="toself",
    mode="lines",
    line=dict(
        width=4,
        color="red",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

'''==============='''
'''    Refined    '''
'''==============='''

fig.add_trace(
  go.Scattergl(
    name="+Ref Pseudo-Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[item for sublist in refpospseudoCyclesX for item in sublist.tolist() + [None]],
    y=[item for sublist in refpospseudoCyclesY for item in sublist.tolist() + [None]],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="rgba(38, 12, 12, 0.2)",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="-Ref Pseudo-Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[item for sublist in refnegpseudoCyclesX for item in sublist.tolist() + [None]],
    y=[item for sublist in refnegpseudoCyclesY for item in sublist.tolist() + [None]],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="rgba(38, 12, 12, 0.2)",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="+Ref PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=np.average(refpospseudoCyclesY, 0),
    # fill="toself",
    mode="lines",
    line=dict(
        width=4,
        color="blue",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

refnegpseudoCyclesY_avg = np.average(refnegpseudoCyclesY, 0)
idx = np.argmax(refnegpseudoCyclesY_avg)
refnegpseudoCyclesY_avg = np.roll(refnegpseudoCyclesY_avg, -idx)

fig.add_trace(
  go.Scattergl(
    name="-Ref PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=refnegpseudoCyclesY_avg,
    # fill="toself",
    mode="lines",
    line=dict(
        width=4,
        color="blue",
        # showscale=False
    ),
    visible = "legendonly"
  )
)


fig.show(config=dict({'scrollZoom': True}))