import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
from scipy.linalg import block_diag
import numpy.polynomial.polynomial as poly
import os, sys
os.chdir('../..')
print (os.path.abspath(os.curdir))

'''==============='''
''' Read wav file '''
'''==============='''

name = "alto"
W, fps = se.read_wav(f"./Python/Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

save_path = os.path.abspath(os.curdir) + '''\\Papers\\02_Sound_Representation\\images\\''' + sys.argv[0].split('/')[-1].replace(".py", "") + "-" + name + ".svg"
print(save_path)

Xpos, Xneg = se.get_frontiers(W)

posL = []
for i in range(1, Xpos.size):
  posL.append(Xpos[i] - Xpos[i - 1])

negL = []
for i in range(1, Xneg.size):
  negL.append(Xneg[i] - Xneg[i - 1])

if np.std(posL) < np.std(negL):
  print("using positive frontier")
  maxL = np.max(np.array(posL))
  pseudoCyclesX = []
  pseudoCyclesY = []
  for i in range(1, Xpos.size):
    x0 = Xpos[i - 1]
    x1 = Xpos[i]
    # a = np.max(np.abs(W[x0 : x1]))
    ft = np.fft.rfft(W[x0 : x1])
    npulse = np.fft.irfft(ft, maxL)
    pseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
    pseudoCyclesX.append(np.arange(maxL))
  pseudoCyclesX = np.array(pseudoCyclesX)
  pseudoCyclesY = np.array(pseudoCyclesY)
else:
  print("using negative frontier")
  maxL = np.max(np.array(negL))
  pseudoCyclesX = []
  pseudoCyclesY = []
  for i in range(1, Xpos.size):
    x0 = Xpos[i - 1]
    x1 = Xpos[i]
    # a = np.max(np.abs(W[x0 : x1]))
    ft = np.fft.rfft(W[x0 : x1])
    npulse = np.fft.irfft(ft, maxL)
    pseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
    pseudoCyclesX.append(np.arange(maxL))
  pseudoCyclesX = np.array(pseudoCyclesX)
  pseudoCyclesY = np.array(pseudoCyclesY)

print(f"Max L = {maxL}")


pseudoCyclesY_avg = np.average(pseudoCyclesY, 0)

'''============================================================================'''
'''                                    APROX                                   '''
'''============================================================================'''

m = pseudoCyclesY_avg.size
# X = np.arange(m, dtype=np.float64)
X = np.linspace(0, 1, m, dtype=np.float64)
print(f"x0 = {X[0]}, xm-1 = {X[m-1]}")
k = 6
Q = np.zeros((m, k + 1))

for i in range(m):
  for j in range(k + 1):
    Q[i, j] = X[i]**j

Q = block_diag(Q, Q)

D = np.eye(k + 1)

D = np.hstack((D, -D))

E = np.zeros((2, 2 * (k + 1)))
for j in range(k + 1):
  E[0, j] = X[0]**j
  E[0, k + j + 1] = -X[m - 1]**j

for j in range(1, k + 1):
  E[1, j] = j * X[0] ** (j - 1)
  E[1, k + j + 1] = -j * X[m - 1] ** (j - 1)

V = np.vstack((D, E))

np.savetxt("V.csv", V, delimiter=",")

Y = np.hstack((pseudoCyclesY_avg, pseudoCyclesY_avg)).T

print(Q.shape, Y.shape)

QTQinv = np.linalg.inv(Q.T @ Q)
tau = np.linalg.inv(V @ QTQinv @ V.T)
QTY = Q.T @ Y
# par1 = V @ QTQinv @ QTY
A = QTQinv @ (QTY - V.T @ tau @ V @ QTQinv @ QTY)
A = np.reshape(A, (2, -1))

Xc = np.linspace(0, 1, 10*m)
Yc = poly.polyval(Xc, A[0, :])


'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''


fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver')

fig.add_trace(
  go.Scatter(
    name="Symmetry line", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[1, 1],
    y=[-1, 1.2],
    # fill="toself",
    mode="lines",
    line=dict(
        width=4,
        color="black",
        dash="dash"
        # showscale=False
    ),
    showlegend=False
    # visible = "legendonly"
  )
)


fig.add_trace(
  go.Scatter(
    name="Pseudo Cycles Average", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=pseudoCyclesY_avg,
    # fill="toself",
    mode="lines",
    line=dict(
        width=4,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Pseudo Cycles Average mirror", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X + 1,
    y=pseudoCyclesY_avg,
    # fill="toself",
    mode="lines",
    line=dict(
        width=4,
        color="gray",
        # dash="dash"
        # showscale=False
    ),
    showlegend=False
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Parametric Pseudo Cycles Average", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xc,
    y=Yc,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="black",
        # shape="spline",
        # smoothing=1
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Parametric Pseudo Cycles Average mirror", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xc + 1,
    y=Yc,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="black",
        # shape="spline",
        # smoothing=1
        # dash="dash"
    ),
    showlegend=False
    # visible = "legendonly"
  )
)



FONT = dict(
  family="Latin Modern Roman",
  color="black",
  size=10
)

fig.update_layout(
  # title = name,
  xaxis_title="$x$",
  yaxis_title="Amplitude",
  # yaxis = dict(scaleanchor = "x", scaleratio = 1 ),                   # <|<|<|<|<|<|<|<|<|<|<|<|
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=FONT
)

fig.show(config=dict({'scrollZoom': True}))
fig.write_image(save_path, width=680, height=400, scale=1, engine="kaleido", format="svg")