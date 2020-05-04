import numpy as np
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#######################
n = 400
t = 5

res_f = 60
res_p = 100
min_f = 15
max_f = 20
min_p = 0
max_p = 2 * np.pi
#######################

res = max(res_f, res_p)

F = np.linspace(0, n, res)
P = np.linspace(0, 2 * np.pi, res)
FP = np.zeros((res, res))

X, Y = [], []
for i in range(res):
  for j in range(res):
    FP[i, j] = np.cos(P[j] + 2*np.pi*F[i]*t/n)
    X.append(P[j])
    Y.append(F[i])

U = np.unique(np.round(FP.flatten('C'), 3))[::-1]
A = np.round(np.cos(np.linspace(0, 2 * np.pi, res)), 3)[0:-1]
print(A.shape[0], U.shape[0])
####

dp = 2 * np.pi / (res - 1)
df = n / (res - 1)
step = max(1, res_f // res_p)

c = int(np.ceil(res / step))
print(len(range(0, res-1, step)), c)

i_f = min((res + 1) // 2, int(np.ceil(max_f * res / n)))
i_i = int(np.floor(min_f * res / n))
print(i_i, i_f)
FPs = np.zeros((i_f - i_i, c))
Xs, Ys, Zs = [], [], []
for i in range(i_i, i_f):
  for j in range(0, res - 1, step):
    idx = (i * t + j) % (res - 1)
    assert(idx >= 0)
    FPs[i - i_i, j // step] = A[idx]
    Xs.append(j * dp)
    Ys.append(i * df)
    Zs.append(A[idx])

try:
  print(np.allclose(np.array(Zs), FPs.flatten('C')))
except:
  print("oops!")
# exit()

####################### DRAW #######################

fig = go.Figure()
fig.update_xaxes(tickvals=P)
fig.update_yaxes(tickvals=F)

fig.add_trace(
  go.Scatter(
    name= "Amplitudes",
    x= X,
    y= Y,
    # legendgroup="Normal",
    customdata=FP.flatten('C'),
    hovertemplate='phase:%{x:.3f}<br>frequency:%{y:.3f}<br>amplitude: %{customdata:.3f} ',
    mode='markers',
    marker=dict(
        size=10,
        color=FP.flatten('C'), #set color equal to a variable
        colorscale='Bluered', # one of plotly colorscales
        showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    name= "Predicted Amplitudes",
    x= Xs,
    y= Ys,
    # legendgroup="Normal",
    customdata=Zs,
    hovertemplate='phase:%{x:.3f}<br>frequency:%{y:.3f}<br>amplitude: %{customdata:.3f} ',
    mode='markers',
    marker_symbol="circle-open",
    marker=dict(
      size=18,
      color=Zs, #set color equal to a variable
      colorscale='Bluered', # one of plotly colorscales
      showscale=False,
      line=dict(
                color=Zs,
                width=4
      )
    )
  )
)

fig.show()