import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp




'''==============='''
''' Read wav file '''
'''==============='''


name = "alto"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)

Xf = np.sort(np.hstack([Xpos, Xneg]))

Ix = hp.split_raw_frontier(Xf, W, 2)
A = hp.constrained_least_squares_arbitrary_intervals(Xf, np.abs(W), Ix, 2)
E = hp.coefs_to_array_arbitrary_intervals(A, Xf, Ix, n)

pa, used_positive_frontier, pcs = hp.pseudocycles_average(Xpos, Xneg, W)

X3d = [] # relative i
Y3d = [] # #pc
Z3d = [] # amp

for i in range(pcs.shape[0]):
  pcs[i] = pcs[i] - pa
  # for j in range(pcs.shape[1]):
  #   X3d.append(j)
  #   Y3d.append(i)
  #   Z3d.append(pcs[i, j])




'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
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
    z=pcs,
    # mode='markers',
    # marker=dict(
    #     size=12,
    #     color=Y3d,                # set color to an array/list of desired values
    #     colorscale='Viridis',   # choose a colorscale
    #     opacity=0.8
    # )
)])

fig.show(config=dict({'scrollZoom': True}))