import numpy as np
import plotly.graph_objects as go

from Helper import read_wav

#######################
#######################
local_res_f = 20
local_res_p = 10
local_min_f = 1
local_max_f = 3
min_p = 0
max_p = 1#2 * np.pi
path = "Python\local_f=2.wav"
#######################
#######################

W, fps = read_wav(path)

n = W.shape[0]

assert min_p < max_p, f"min_p={round(min_p, 3)} >= max_p={round(max_p, 3)}"

assert local_min_f < local_max_f, f"local_min_f={round(local_min_f, 3)} >= local_max_f={round(local_max_f, 3)}"

preliminar_df = (local_max_f - local_min_f) / local_res_f
global_res_f = int(np.ceil(n / preliminar_df))

preliminar_dp = (max_p - min_p) / local_res_p
global_res_p = int(np.ceil(2 * np.pi / preliminar_dp))

res = max(global_res_f, global_res_p)

print(f"n={n} | Global Resolution f={global_res_f} | Global Resolution p={global_res_p} | Resolution={res}")

A = np.round(np.cos(np.linspace(0, 2 * np.pi, res)), 3)[0:-1]

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
FP = np.zeros((l , c))


# Xs, Ys = [], []
for t in range(n):
  for i in range(l):
    for j in range(c):
      idx = ((i_i + i * i_s) * t + j_i + j * j_s) % (res - 1)
      # assert(idx >= 0)
      FP[i, j] += A[idx] * W[t]

FP = FP * 2 / n

ind_f, ind_p = np.unravel_index(np.argmax(FP, axis=None), FP.shape)

max_p = np.round((j_i + ind_p) * dp, 2)
max_f = np.round((i_i + ind_f) * df, 2)

print(f"f={max_f}, p={max_p}")

####################### DRAW #######################

fig = go.Figure()

X = np.arange(j_i, j_f, j_s) * dp
Y = np.arange(i_i, i_f, i_s) * df
fig.add_trace(go.Heatmap(z=FP, x=X, y=Y))

fig.update_xaxes(tickvals=X)
fig.update_yaxes(tickvals=Y)

fig.show()