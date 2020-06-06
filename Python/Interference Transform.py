import numpy as np
import plotly.graph_objects as go
import random
from Helper import read_wav

#######################
#######################
n = 10
res_f = 20
res_p = 10
min_f = 1
max_f = 3
min_p = 0
max_p = 1#2 * np.pi
#######################
#######################

### Generating random wave
random.seed(1)
X = np.arange(n)
W = np.zeros(n)
number_of_random_waves = 5
A = np.array([random.uniform(1, 5) for i in range(number_of_random_waves)])
F = np.array([random.uniform(1, 9) for i in range(number_of_random_waves)])
P = np.array([random.uniform(0, 2 * np.pi) for i in range(number_of_random_waves)])
W = np.sum(A * np.cos(P + 2 * np.pi * F.T * X[:, np.newaxis] / n), 1)
W = W / np.max(np.abs(W))
idx = np.argmax(A)

# print(W.shape)
# exit()



assert min_p < max_p, f"min_p={round(min_p, 3)} >= max_p={round(max_p, 3)}"

assert min_f < max_f, f"local_min_f={round(min_f, 3)} >= local_max_f={round(max_f, 3)}"

preliminar_df = (max_f - min_f) / res_f
global_res_f = int(np.ceil(n / preliminar_df))

preliminar_dp = (max_p - min_p) / res_p
global_res_p = int(np.ceil(2 * np.pi / preliminar_dp))

res = max(global_res_f, global_res_p)

print(f"n={n} | Global Resolution f={global_res_f} | Global Resolution p={global_res_p} | Resolution={res}")

A = np.round(np.cos(np.linspace(0, 2 * np.pi, res)), 3)[0:-1]

dp = (2 * np.pi) / (res - 1)
df = n / (res - 1)

f_idx_ini = int(np.round(min_f / df))
f_idx_fin = int(np.round(max_f / df))
f_step = max(1, int(np.round((f_idx_fin - f_idx_ini) / res_f)))

p_idx_ini = int(np.round(min_p / dp))
p_idx_fin = int(np.round(max_p / dp))
p_step = max(1, int(np.round((p_idx_fin - p_idx_ini) / res_p)))

print(f"i_i={f_idx_ini}, i_f={f_idx_fin}, i_s={f_step} | j_i={p_idx_ini}, j_f={p_idx_fin}, j_s={p_step}")

l = int(np.round((f_idx_fin - f_idx_ini) / f_step))
c = int(np.round((p_idx_fin - p_idx_ini) / p_step))
print(f"Lines={l}, Columns={c}")
FP = np.zeros((l , c))


# Xs, Ys = [], []
for t in range(n):
  for i in range(l):
    for j in range(c):
      idx = ((f_idx_ini + i * f_step) * t + p_idx_ini + j * p_step) % (res - 1)
      # assert(idx >= 0)
      FP[i, j] += A[idx] * W[t]

FP = FP * 2 / n

ind_f, ind_p = np.unravel_index(np.argmax(FP, axis=None), FP.shape)

max_p = np.round((p_idx_ini + ind_p * p_step) * dp, 2)
max_f = np.round((f_idx_ini + ind_f * f_step) * df, 2)

print(f"f={max_f}, p={max_p}")

####################### DRAW #######################

fig = go.Figure()

X = np.arange(p_idx_ini, p_idx_fin, p_step) * dp
Y = np.arange(f_idx_ini, f_idx_fin, f_step) * df
fig.add_trace(go.Heatmap(z=FP, x=X, y=Y))

fig.update_xaxes(tickvals=X)
fig.update_yaxes(tickvals=Y)

fig.show()