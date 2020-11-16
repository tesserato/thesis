import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp
from statistics import mode



'''==============='''
''' Read wav file '''
'''==============='''


name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)
Xpos = hp.refine_frontier_iter(Xpos, W)
Xneg = hp.refine_frontier_iter(Xneg, W)

T = []
for i in range(1, Xpos.size):
  T.append(Xpos[i] - Xpos[i - 1])
for i in range(1, Xneg.size):
  T.append(Xneg[i] - Xneg[i - 1])
T = np.array(T, dtype = np.int)
maxT = np.max(T)
modeT = mode(T)

Xf = np.sort(np.hstack([Xpos, Xneg]))

Ix = hp.split_raw_frontier(Xf, W, 2)
A = hp.constrained_least_squares_arbitrary_intervals(Xf, np.abs(W), Ix, 2)
E = hp.coefs_to_array_arbitrary_intervals(A, Xf, Ix, n)

posft, pospcs, negft, negpcs = hp.average_pc_waveform_return_pcsandfts(Xpos, Xneg, W)

# pospc = []
# for ft in posft:
#   pospc.append(np.fft.ifft(ft))
# pospc = np.array(pospc)

# X3d = [] # relative i
# Y3d = [] # #pc
# Z3d = [] # amp

# for i in range(pospcs.shape[0]):
#   pospcs[i] = pospcs[i] - pa
  # for j in range(pcs.shape[1]):
  #   X3d.append(j)
  #   Y3d.append(i)
  #   Z3d.append(pcs[i, j])




'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
# fig = go.Figure()
# fig.layout.template ="plotly_white"
# # fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
# fig.update_layout(
#   # title = name,
#   xaxis_title="x",
#   yaxis_title="Amplitude",

#   # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
#   #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
#   #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
#   # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

#   legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
#   margin=dict(l=5, r=5, b=5, t=5),
#   font=dict(
#   family="Computer Modern",
#   color="black",
#   size=18
#   )
# )

fig = go.Figure(data=[go.Surface(
    # x=X3d,
    # y=Y3d,
    z=pospcs,
    # mode='markers',
    # marker=dict(
    #     size=12,
    #     color=Y3d,                # set color to an array/list of desired values
    #     colorscale='Viridis',   # choose a colorscale
    #     opacity=0.8
    # )
)])

fig.show(config=dict({'scrollZoom': True}))

fig = go.Figure(data=[go.Surface(
    # x=X3d,
    # y=Y3d,
    z=np.abs(posft)[:, :10],
    # mode='markers',
    # marker=dict(
    #     size=12,
    #     color=Y3d,                # set color to an array/list of desired values
    #     colorscale='Viridis',   # choose a colorscale
    #     opacity=0.8
    # )
)])

fig.show(config=dict({'scrollZoom': True}))