import numpy as np
from scipy import interpolate
from statistics import mode
import signal_envelope as se
import plotly.graph_objects as go
import helper as hp # <|<|<|


def average_pc_waveform(Xp, W):
  # amp = np.max(np.abs(W))
  max_T = int(np.max(np.abs(Xp[1:] - Xp[:-1])))
  Xlocal = np.linspace(0, 1, max_T)
  orig_pcs = []
  norm_pcs = []
  for i in range(1, Xp.size):
    x0 = Xp[i - 1]
    x1 = Xp[i] + 1
    orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, W[x0 : x1].size), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      # Ylocal = Ylocal / np.max(np.abs(Ylocal)) * amp
      norm_pcs.append(Ylocal)  
  return np.average(np.array(norm_pcs), 0), orig_pcs, norm_pcs

def refine_Xpc_alt(W, avgpc, min_size, max_size):
  pcx = interpolate.interp1d(np.linspace(0, 1, avgpc.size), avgpc / np.max(np.abs(avgpc)), "linear")

  W = np.pad(W, [min_size, 0])

  fig.add_trace(
    go.Scattergl(
      name="w", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=X,
      y=W,
      # fill="toself",
      mode="lines",
      line=dict(
          width=1,
          color="black",
          # showscale=False
      ),
      # visible = "legendonly"
    )
  )

  Xpc_new = []
  sizes = np.arange(min_size, max_size, 1)
  # posit = np.arange(min_size - 1)
  posit = np.arange(W.size - max_size)
  idxs = np.zeros((sizes.size, posit.size))
  cycles = dict()
  for size in sizes:
    # print(f"size = {size}:")
    pc = pcx(np.linspace(0, 1, size, endpoint=False))
    cycles[size] = pc
    for x0 in posit:
      w = W[x0 : x0 + size] # / np.max(np.abs(W[x0 : x0 + size]))
      idxs[size - min_size, x0] = np.average(np.sign(pc) * np.sign(w))
      # print(f"x0={x0}, {idxs[size, x0]}")
  opt_size, opt_x0 = np.unravel_index(idxs.argmax(), idxs.shape)
  print(f"opt_size idx = {opt_size}, opt_x0 idx = {opt_x0} |||| opt_size = {opt_size + min_size} of max = {max_size}")
  
  fig.add_trace(
    go.Scattergl(
      name="", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=np.arange(opt_size + min_size) + opt_x0,
      y=pcx(np.linspace(0, 1, opt_size + min_size, endpoint=False)),
      showlegend=False,
      mode="lines",
      line=dict(
          width=1,
          color="blue",
          # showscale=False
      ),
      # visible = "legendonly"
    )
  )

  Xpc_new.append(opt_x0)
  Xpc_new.append(opt_x0 + opt_size + min_size)
  
  x0 = Xpc_new[-1]
  opt_size = None
  while x0 + max_size < W.size:
    max_conv = 0
    for size, pc in cycles.items():
      # pc = pcx(np.linspace(0, 1, size, endpoint=False))
      w = W[x0 : x0 + size] #/ np.max(np.abs(W[x0 : x0 + size]))
      conv = np.average(pc * w)
      if conv > max_conv:
        opt_size = size
        max_conv = conv
    # if opt_size == None:
      # break
    fig.add_trace(
      go.Scattergl(
        name="", # <|<|<|<|<|<|<|<|<|<|<|<|
        x=np.arange(opt_size) + x0,
        y=pcx(np.linspace(0, 1, opt_size, endpoint=False)),
        showlegend=False,
        mode="lines",
        line=dict(
            width=1,
            color="red",
            # showscale=False
        ),
        # visible = "legendonly"
      )
    )
    x0 += opt_size
    Xpc_new.append(x0)
  Xpc_new = np.array(Xpc_new, np.int) - min_size
  Xpc_new = np.unique(Xpc_new[Xpc_new > 0])
  return Xpc_new[Xpc_new < W.size].astype(np.int)

def to_plot(Matrix):
  X = []
  Y = []
  for line in Matrix:
    for x, y in enumerate(line):
      X.append(x)
      Y.append(y)
    X.append(None)
    Y.append(None)
  return X, Y

transp_black = "rgba(38, 12, 12, 0.2)"

fig = go.Figure()


'''###### Read wav file ######'''

name = "tom"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
W = W / amp
n = W.size

Xpos, Xneg = se.get_frontiers(W)
Xpos = hp.refine_frontier_iter(Xpos, W).astype(np.int)
Xneg = hp.refine_frontier_iter(Xneg, W).astype(np.int)

n_Xpc = min(Xpos.size, Xneg.size)
Xpc = np.round((Xpos[:n_Xpc] + Xneg[:n_Xpc]) / 2).astype(np.int)
Xpc = np.unique((Xpc[Xpc > 0]))

L = Xpc[1:] - Xpc[:-1]
mdeL = mode(L)
stdL = np.average(np.abs(L - mdeL))
mult = 4

# min_size = max(np.min(L) - int(stdL * mult), 3)
# max_size = np.max(L) + int(stdL * mult)

# min_size = max(mdeL - int(stdL * mult), 3)
# max_size = mdeL + int(stdL * mult)

min_size = int(mdeL * 0.7)
max_size = int(mdeL * 1.3)

print(f"mode={mdeL}, min_size={min_size}, max_size={max_size}")

avgpc, orig_pcs, norm_pcs = average_pc_waveform(Xpc, W)
# avgpc = np.cos(np.linspace(0, 2 * np.pi, avgpc.size))

Xpc_1 = refine_Xpc_alt(W, avgpc, min_size, max_size)

# avgpc_1, orig_pcs_1, norm_pcs_1 = average_pc_waveform(Xpc_1, W)
# Xpc_2 = refine_Xpc_alt(Xpc_1, W, avgpc_1)

# avgpc_2, orig_pcs_2, norm_pcs_2 = average_pc_waveform(Xpc_2, W)
# Xpc_3 = refine_Xpc_alt(Xpc_2, W, avgpc_2)

# avgpc_3, orig_pcs_3, norm_pcs_3 = average_pc_waveform(Xpc_3, W)

fig.layout.template ="plotly_white"
fig.update_yaxes(zeroline=False)#, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="Length",
  yaxis_title="Number of Ocurrences",

  # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
  # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=12
  )
)

# fig.add_trace(
#   go.Scattergl(
#     name="avgpc", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=X,
#     y=avgpc,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scattergl(
#     name="avgpc 1", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=X,
#     y=avgpc_1,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scattergl(
#     name="avgpc 2", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=X,
#     y=avgpc_2,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scattergl(
#     name="avgpc 3", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=X,
#     y=avgpc_3,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color="black",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

# X_local, Y_local = to_plot(norm_pcs_3)
# fig.add_trace(
#   go.Scattergl(
#     name="norm_pcs_3", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=X_local,
#     y=Y_local,
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=1,
#         color=transp_black,
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

fig.show()