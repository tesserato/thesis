# from os import linesep
from statistics import mode
import numpy as np
import plotly.graph_objects as go
import signal_envelope as se
from scipy import interpolate
import sys


def get_pcs(Xpc, W):
  Xpc = Xpc.astype(int)
  # amp = np.max(np.abs(W))
  # max_T = int(np.max(np.abs(Xpc[1:] - Xpc[:-1])))
  Xlocal = np.linspace(0, 1, mode(Xpc[1:] - Xpc[:-1]))
  orig_pcs = []
  norm_pcs = []
  for i in range(2, Xpc.size):
    x0 = Xpc[i - 1]
    x1 = Xpc[i] + 1
    orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1-x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      # Ylocal = Ylocal / np.max(np.abs(Ylocal)) * amp
      norm_pcs.append(Ylocal)
  return np.average(np.array(norm_pcs), 0), orig_pcs, norm_pcs

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

name = "cello"

'''###### Read wav file ######'''
W, fps = se.read_wav(f"./original_samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
W = W / amp
n = W.size

Xpc0 = se.get_frontiers(W)

'''###### Read Pseudo-cycles info ######'''
Xpc = np.genfromtxt(f"./csvs/{name}.csv", delimiter=",")

# X_xpcs, Y_xpcs = [], []
# for x in Xpc:
#   X_xpcs.append(x)
#   X_xpcs.append(x)
#   X_xpcs.append(None)
#   Y_xpcs.append(-1)
#   Y_xpcs.append(1)
#   Y_xpcs.append(None)


average_waveform, orig_waveforms, norm_waveforms = get_pcs(Xpc, W)

T = (Xpc[1 :] - Xpc[: -1]).astype(np.int)

# W = np.pad(W, mode(T))


'''============================================================================'''

FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

'''============================================================================'''
'''                               PLOT WAVEFORM                                '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="<b>Frame</b>",
  yaxis_title="<b>Amplitude</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='gray', range=[-100, 317])
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='gray', range=[-0.8, 0.8])

s = 10
r = average_waveform.size + s + 1

W = np.pad(W, r)

fig.add_trace(
  go.Scatter(
    name="Signal      ", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=np.arange(W.size) - r,
    y=W,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Translated average waveforms      ", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=np.arange(W.size) - r,
    y=[0],
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

mult = []
for i in range(0, r + 11, s):
  mult.append(np.average(W[i : i + average_waveform.size] * average_waveform))

mult = np.array(mult)

mult = mult - np.min(mult)

mult = mult / np.max(np.abs(mult))

for i in range(0, r + 11, s):
  val = np.sign(mult[int(i / s)]) * mult[int(i / s)]**2
  wid = 1
  print(i, val)
  color = f"rgba(255, 0, 0, {round(val, 4)})"

  if val == 1.0:
    wid = 2
  fig.add_trace(
    go.Scatter(
      name=f"{i}", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=np.arange(average_waveform.size) - r + i,
      y=average_waveform,
      # fill="toself",
      mode="lines",
      line=dict(
          width=wid,
          color=color,
          # showscale=False
      ),
      showlegend=False
      # visible = "legendonly"
    )
  )



# fig.show(config=dict({'scrollZoom': True}))
save_name = sys.argv[0].split('/')[-1].replace(".py", "")
wid = 650
hei = 200
fig.write_image("./paper/images/" + save_name + ".svg", width=wid, height=hei, engine="kaleido", format="svg")
fig.write_image("./site/public/images/" + save_name + ".webp", width=int(1.7*wid), height=int(1.5*hei), format="webp")
fig.write_html("./site/public/images/" + save_name + ".html", include_plotlyjs="cdn", include_mathjax="cdn")
print("saved:", save_name)

