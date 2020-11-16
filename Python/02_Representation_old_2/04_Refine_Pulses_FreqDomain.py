import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp
from statistics import mode
from scipy import interpolate

def normalize_pcs(Xpcs, W, maxT):
  nft = int(maxT//2 + 1)
  Xlocal = np.linspace(0, 1, maxT)
  ft = np.zeros(nft, dtype=np.complex)
  for i in range(1, Xpcs.size):
    x0 = Xpcs[i - 1]
    x1 = Xpcs[i]
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1 - x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      ft += np.fft.rfft(Ylocal)

  # ft[0] = 0
  f = np.argmax(np.abs(ft))
  p = np.angle(ft[f])
  print(f"f={f}, p={p}")
  ####### SHIFT #######
  Y = np.fft.irfft(ft, maxT)
  d = maxT - np.argmax(Y)
  x = 2 * np.pi * np.arange(nft) / maxT
  ft = np.exp(-1j * x * d) * ft
  #####################
  Y = np.fft.irfft(ft, maxT)
  Y = Y / np.max(np.abs(Y))
  return Y

# def normalize_pcs(Xpcs, W, maxT):
#   ft = np.zeros(int(maxT//2 + 1), dtype=np.complex)
#   for i in range(1, Xpcs.size):
#     x0 = Xpcs[i - 1]
#     x1 = Xpcs[i]
#     slack = maxT - (x1 - x0)
#     left = int(slack // 2)
#     right = int(slack - left)
#     w = np.pad(W[x0 : x1], (left, right), 'constant', constant_values=(0, 0))
#     ft += np.fft.rfft(w)
#   Y = np.fft.irfft(ft, maxT)
#   Y = Y / np.max(np.abs(Y))
#   return Y

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

L = []
for i in range(1, Xpos.size):
  L.append(Xpos[i] - Xpos[i - 1])
for i in range(1, Xneg.size):
  L.append(Xneg[i] - Xneg[i - 1])
L = np.array(L, dtype = np.int)
maxL = np.max(L)
avgL = mode(L)
print(f"Max L = {maxL}")

Xpc = [i for i in range(maxL)]

pospseudoCyclesY = normalize_pcs(Xpos, W, avgL)

negpseudoCyclesY = normalize_pcs(Xneg, W, avgL)
# idx = np.argmax(negpseudoCyclesY)
# negpseudoCyclesY = np.roll(negpseudoCyclesY, -idx)


'''==============='''
'''    Refined    '''
'''==============='''
Xpos = hp.refine_frontier_iter(Xpos, W)
Xneg = hp.refine_frontier_iter(Xneg, W)

print("Sizes:", Xpos.size, Xneg.size)

refpospseudoCyclesY = normalize_pcs(Xpos, W, avgL)

refnegpseudoCyclesY = normalize_pcs(Xneg, W, avgL)
# idx = np.argmax(refnegpseudoCyclesY)
# refnegpseudoCyclesY = np.roll(refnegpseudoCyclesY, -idx)


# normaveragerefpospseudoCyclesY = averagerefpospseudoCyclesY - np.average(averagerefpospseudoCyclesY)
# normaveragerefpospseudoCyclesY = normaveragerefpospseudoCyclesY / np.max(np.abs(normaveragerefpospseudoCyclesY))

# normaveragerefnegpseudoCyclesY = averagerefnegpseudoCyclesY - np.average(averagerefnegpseudoCyclesY)
# normaveragerefnegpseudoCyclesY = normaveragerefnegpseudoCyclesY / np.max(np.abs(normaveragerefnegpseudoCyclesY))


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
    y=pospseudoCyclesY,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="black",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="-Pseudo-Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pospseudoCyclesY for item in Xpc + [None]],
    y=negpseudoCyclesY,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="black",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scattergl(
#     name="+PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
#     y=averagepospseudoCyclesY,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=4,
#         color="red",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

# # negpseudoCyclesY_avg = np.average(negpseudoCyclesY, 0)

# fig.add_trace(
#   go.Scattergl(
#     name="-PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
#     y=averagenegpseudoCyclesY,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=4,
#         color="red",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

'''==============='''
'''    Refined    '''
'''==============='''

fig.add_trace(
  go.Scattergl(
    name="+Ref Pseudo-Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in refpospseudoCyclesY for item in Xpc + [None]],
    y=refpospseudoCyclesY,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="blue",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="-Ref Pseudo-Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in refnegpseudoCyclesY for item in Xpc + [None]],
    y=refnegpseudoCyclesY,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="blue",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scattergl(
    name="PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=(refpospseudoCyclesY + refnegpseudoCyclesY) / 2,
    # fill="toself",
    mode="lines",
    line=dict(
        width=4,
        color="black",
        # showscale=False
    ),
    visible = "legendonly"
  )
)

# # refnegpseudoCyclesY_avg = np.average(refnegpseudoCyclesY, 0)

# fig.add_trace(
#   go.Scattergl(
#     name="-Ref PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
#     y=averagerefnegpseudoCyclesY,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=4,
#         color="blue",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scattergl(
#     name="+Norm Ref PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
#     y=normaveragerefpospseudoCyclesY,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=4,
#         color="green",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

# # refnegpseudoCyclesY_avg = np.average(refnegpseudoCyclesY, 0)

# fig.add_trace(
#   go.Scattergl(
#     name="-Norm Ref PC Avg", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
#     y=normaveragerefnegpseudoCyclesY,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=4,
#         color="green",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )


fig.show(config=dict({'scrollZoom': True}))