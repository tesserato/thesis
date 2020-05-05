import numpy as np
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#######################
#######################
n = 40
t = 36
local_res_f = 20
local_res_p = 20
local_min_f = 0
local_max_f = 20
min_p = 0
max_p = 2 * np.pi
#######################
#######################

assert min_p < max_p, f"min_p={round(min_p, 3)} >= max_p={round(max_p, 3)}"

assert local_min_f < local_max_f, f"local_min_f={round(local_min_f, 3)} >= local_max_f={round(local_max_f, 3)}"

preliminar_df = (local_max_f - local_min_f) / local_res_f
global_res_f = int(np.ceil(n / preliminar_df))

preliminar_dp = (max_p - min_p) / local_res_p
global_res_p = int(np.ceil(2 * np.pi / preliminar_dp))

res = max(global_res_f, global_res_p)

print(f"n={n} | Global Resolution f={global_res_f} | Global Resolution p={global_res_p} | Resolution={res}")

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
print(f"Unique amplitudes={U.shape[0]} | Predicted Sinusoid Resolution={res-1} | Sinusoid Resolution={A.shape[0]}")

try:
  print(f"{np.allclose(U, A[:(res + 1)//2])}: Predicted sinusoid == Calculated sinusoid")
except:
  print(f"FALSE: Predicted sinusoid != Calculated sinusoid")

######

dp = (2 * np.pi) / (res - 1)
df = n / (res - 1)

i_i = int(np.round(local_min_f / df))
i_f = int(np.round(local_max_f / df))
i_s = max(1, int(np.round((i_f - i_i) / local_res_f)))

j_i = int(np.round(min_p / dp))
j_f = int(np.round(max_p / dp))
j_s = max(1, int(np.round((j_f - j_i) / local_res_p)))

print(f"i_i={i_i}, i_f={i_f}, i_s={i_s} | j_i={j_i}, j_f={j_f}, j_s={j_s}")

l = int(np.round((i_f - i_i) / i_s))
c = int(np.round((j_f - j_i) / j_s))
print(f"Lines={l}, Columns={c}")
FPs = np.zeros((l , c))
Xs, Ys = [], []
for i in range(l):
  for j in range(c):
    idx = ((i_i + i * i_s) * t + j_i + j * j_s) % (res - 1)
    assert(idx >= 0)
    FPs[i, j] = A[idx]
    # print(f"idxi={i} | idxj={j} | idx={idx}")
    Xs.append((j_i + j * j_s) * dp)
    Ys.append((i_i + i * i_s) * df)

naive = np.round(FP[i_i : i_f : i_s , j_i : j_f : j_s], 3)
try:
  print(f"{np.allclose(FPs, naive)}: Naive == Symmetries")
except:
  print(f"FALSE: Naive != Symmetries")
print(f"Naive shape={naive.shape} | Symmetries shape={FPs.shape}")

####################################################
####################### DRAW #######################
####################################################

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
  ))

fig.add_trace(
  go.Scatter(
    name= "Predicted Amplitudes",
    x= Xs,
    y= Ys,
    # legendgroup="Normal",
    customdata=FPs.flatten('C'),
    hovertemplate='phase:%{x:.3f}<br>frequency:%{y:.3f}<br>amplitude: %{customdata:.3f} ',
    mode='markers',
    marker_symbol="circle-open",
    marker=dict(
      size=18,
      color=FPs.flatten('C'), #set color equal to a variable
      colorscale='Bluered', # one of plotly colorscales
      showscale=False,
      line=dict(
                color=FPs.flatten('C'),
                width=4
      )
    )
  ))

fig.add_trace(
  go.Scatter(
    name= "Lines",
    x= [0, 2 * np.pi, None, 0, 2 * np.pi, None, min_p, min_p, None, max_p, max_p],
    y= [local_min_f, local_min_f, None, local_max_f, local_max_f, None, 0, n, None, 0, n],
    customdata=FP.flatten('C'),
    hovertemplate='phase:%{x:.3f}<br>frequency:%{y:.3f}<br>amplitude: %{customdata:.3f} ',
    mode='lines',
    line=go.scatter.Line(color="gray")
  ))

fig.show()