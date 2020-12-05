'''To be used AFTER signal_envelope is installed via pip'''
import sys
# sys.path.append("signal_envelope/")
import numpy as np
import signal_envelope as se
from scipy import interpolate
import plotly.graph_objects as go

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

name = "alto"
W, _ = se.read_wav(f"C:/Users/tesse/Desktop/Files/Dropbox/0_Science/reps/envelope/test_samples/{name}.wav")
amp = np.max(np.abs(W))
W = 4 * W / amp
X = np.arange(W.size)

Xpos, Xneg = se.get_frontiers(W, 0)
E = se.get_frontiers(W, 1)

m = min(Xpos.size, Xneg.size)
Xavg = np.round((Xpos[0 : m] + Xneg[0 : m]) / 2).astype(np.int)

avg_pcs, orig_pcs, norm_pcs = average_pc_waveform(Xavg, W)

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''

FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

'''Plotting'''
fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="<b><i>i</i></b>",
  yaxis_title="<b>Amplitude</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zeroline=False)

XX, YY = to_plot(norm_pcs)
fig.add_trace(
  go.Scatter(
    name="Normalized Pseudo Cycles    ",
    x=XX,
    y=YY,
    mode="lines",
    # fill="tozeroy",
    # fillcolor="rgba(0,0,0,0.3)",
    line=dict(width=1, color="rgba(235, 64, 52,0.2)",),
  )
)

fig.add_trace(
  go.Scatter(
    name="Average Waveform    ",
    # x=np.arange(W.size),
    y=avg_pcs,
    mode="lines",
    # fill="tozeroy",
    # fillcolor="rgba(0,0,0,0.6)",
    line=dict(width=3, color="black",),
  )
)

fig.add_trace(
  go.Scatter(
    name="Average Waveform (Transposed)    ",
    x=avg_pcs.size + np.arange(avg_pcs.size) - 1,
    y=avg_pcs,
    mode="lines",
    # fill="tozeroy",
    # fillcolor="rgba(0,0,0,0.6)",
    line=dict(width=3, color="black", dash="dot"),
  )
)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", "") + ".svg"
fig.write_image(save_name, width=650, height=260, engine="kaleido", format="svg")
print("saved:", save_name)
