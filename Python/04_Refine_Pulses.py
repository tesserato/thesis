import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp

def normalize_pcs(Xpcs, W, maxT):
  weights = []
  pseudoCyclesY = []
  for i in range(1, Xpcs.size):
    x0 = Xpcs[i - 1]
    x1 = Xpcs[i]
    weights.append(np.sum(np.abs(W[x0 : x1])))
    ft = np.fft.rfft(W[x0 : x1])
    npulse = np.fft.irfft(ft, maxT)
    pseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
    # pospseudoCyclesX.append(Xpc)
  # pospseudoCyclesX = np.array(pospseudoCyclesX)
  return np.array(pseudoCyclesY), weights

'''==============='''
''' Read wav file '''
'''==============='''

name = "alto"
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

Xpc = [i for i in range(maxL)]

pospseudoCyclesY, posW = normalize_pcs(Xpos, W, maxL)
averagepospseudoCyclesY = np.average(pospseudoCyclesY, 0, posW)

negpseudoCyclesY, negW = normalize_pcs(Xneg, W, maxL)
averagenegpseudoCyclesY = np.average(negpseudoCyclesY, 0, negW)

idx = np.argmax(averagenegpseudoCyclesY)
averagenegpseudoCyclesY = np.roll(averagenegpseudoCyclesY, -idx)


'''==============='''
'''    Refined    '''
'''==============='''
Xpos = hp.refine_frontier_iter(Xpos, W)

Xneg = hp.refine_frontier_iter(Xneg, W)

print("Sizes:", Xpos.size, Xneg.size)

refpospseudoCyclesY, posW = normalize_pcs(Xpos, W, maxL)
averagerefpospseudoCyclesY = np.average(refpospseudoCyclesY, 0, posW)

refnegpseudoCyclesY, negW = normalize_pcs(Xneg, W, maxL)
averagerefnegpseudoCyclesY = np.average(refnegpseudoCyclesY, 0, negW)

idx = np.argmax(averagerefnegpseudoCyclesY)
averagerefnegpseudoCyclesY = np.roll(averagerefnegpseudoCyclesY, -idx)

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
    x=[item for sublist in pospseudoCyclesY for item in Xpc + [None]],
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
    x=[item for sublist in pospseudoCyclesY for item in Xpc + [None]],
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
    y=averagepospseudoCyclesY,
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

# negpseudoCyclesY_avg = np.average(negpseudoCyclesY, 0)

fig.add_trace(
  go.Scattergl(
    name="-PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=averagenegpseudoCyclesY,
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
    x=[item for sublist in refpospseudoCyclesY for item in Xpc + [None]],
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
    x=[item for sublist in refnegpseudoCyclesY for item in Xpc + [None]],
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
    y=averagerefpospseudoCyclesY,
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

# refnegpseudoCyclesY_avg = np.average(refnegpseudoCyclesY, 0)

fig.add_trace(
  go.Scattergl(
    name="-Ref PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=averagerefnegpseudoCyclesY,
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