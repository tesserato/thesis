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
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)

Xf = np.sort(np.hstack([Xpos, Xneg]))

# Ix = hp.split_raw_frontier(Xf, W, 2)
# A = hp.constrained_least_squares_arbitrary_intervals(Xf, np.abs(W), Ix, 2)
# E = hp.coefs_to_array_arbitrary_intervals(A, Xf, Ix, n)

pa, used_positive_frontier, pcs = hp.pseudocycles_average(Xpos, Xneg, W)

for i in range(pcs.shape[0]):
  pcs[i] = pa

P = pcs.flatten()

# Q = np.zeros((P.size - 3, 7))
# Y = np.zeros((P.size - 3))
# X = np.zeros((P.size - 3, 2))
# for i in range(P.size - 3):
#   X[i, 0] = P[i]
#   X[i, 1] = P[i + 1]

#   Q[i, 0] = 1
#   Q[i, 1] = P[i]
#   Q[i, 2] = P[i]**2
#   Q[i, 3] = P[i]**3

#   # Q[i, 4] = 1
#   Q[i, 4] = P[i + 1]
#   Q[i, 5] = P[i + 1]**2
#   Q[i, 6] = P[i + 1]**3

#   Y[i] = P[i + 2]

# # np.savetxt("Q.csv", Q[], delimiter=",")

# A = np.linalg.inv(Q.T @ Q) @ Q.T @ Y

# x0 = P[0]
# x1 = P[1]
# Wp = []
# for i in range(n):
#   x = A[0] + A[1] * x0 + A[2] * x0**2 + A[3] * x0**3 + A[4] * x1 + A[5] * x1**2 + A[6] * x1**3
#   Wp.append(x)
#   x0 = x1
#   x1 = x

# Wp = np.array(Wp)
# Wp = Wp / np.max(np.abs(Wp)) * amp

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white"
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",
  # yaxis = dict(scaleanchor = "x", scaleratio = 1),                   # <|<|<|<|<|<|<|<|<|<|<|<|
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=P,
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Pred Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=X,
#     y=Wp,
#     mode="lines",
#     line=dict(
#         # size=8,
#         color="red",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )


fig.show(config=dict({'scrollZoom': True}))