from ctypes import resize
import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import sys
import os
os.chdir('../..')
print (os.path.abspath(os.curdir))
from plotly.subplots import make_subplots

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

# exit()

Xpos, Xneg = se.get_frontiers(W)

maxL = []
for i in range(1, Xpos.size):
  maxL.append(Xpos[i] - Xpos[i - 1])
for i in range(1, Xneg.size):
  maxL.append(Xneg[i] - Xneg[i - 1])
maxL = np.max(np.array(maxL))
print(f"Max L = {maxL}")

pospseudoCyclesX = []
pospseudoCyclesY = []
for i in range(1, Xpos.size):
  x0 = Xpos[i - 1]
  x1 = Xpos[i]
  # a = np.max(np.abs(W[x0 : x1]))
  ft = np.fft.rfft(W[x0 : x1])
  npulse = np.fft.irfft(ft, maxL)
  pospseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
  pospseudoCyclesX.append(np.arange(maxL))
pospseudoCyclesX = np.array(pospseudoCyclesX)
pospseudoCyclesY = np.array(pospseudoCyclesY)

negpseudoCyclesX = []
negpseudoCyclesY = []
for i in range(1, Xneg.size):
  x0 = Xneg[i - 1]
  x1 = Xneg[i]
  # a = np.max(np.abs(W[x0 : x1]))
  ft = np.fft.rfft(W[x0 : x1])
  npulse = np.fft.irfft(ft, maxL)
  negpseudoCyclesY.append(npulse / np.max(np.abs(npulse)))
  negpseudoCyclesX.append(np.arange(maxL))
negpseudoCyclesX = np.array(negpseudoCyclesX)
negpseudoCyclesY = np.array(negpseudoCyclesY)



'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="$\\text{Relative} \\ i$",
  yaxis_title="Amplitude",
  # yaxis = dict(scaleanchor = "x", scaleratio = 1 ),                   # <|<|<|<|<|<|<|<|<|<|<|<|
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Latin Modern Roman",
  color="black",
  size=10
  )
)

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver')

fig.add_trace(
  go.Scatter(
    name="Normalized pseudo-cycles (positive frontier)", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[item for sublist in pospseudoCyclesX for item in sublist.tolist() + [None]],
    y=[item for sublist in pospseudoCyclesY for item in sublist.tolist() + [None]],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="rgba(38, 12, 12, 0.1)",
        # showscale=False
    ),
    # visible = "legendonly"
  ),row=1, col=1
)

fig.add_trace(
  go.Scatter(
    name="Normalized pseudo-cycles (negative frontier)", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[item for sublist in negpseudoCyclesX for item in sublist.tolist() + [None]],
    y=[item for sublist in negpseudoCyclesY for item in sublist.tolist() + [None]],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="rgba(38, 12, 12, 0.1)",
        # showscale=False
    ),
    # visible = "legendonly"
  ),row=1, col=2
)

fig.add_trace(
  go.Scatter(
    name="Average normalized pseudo-cycle (positive frontier)", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=np.average(pospseudoCyclesY, 0),
    # fill="toself",
    mode="lines",
    line=dict(
        width=3,
        color="silver",
        # showscale=False
    ),
    # visible = "legendonly"
  ),row=1, col=1
)


negpseudoCyclesY_avg = np.average(negpseudoCyclesY, 0)
# idx = np.argmax(negpseudoCyclesY_avg)
# negpseudoCyclesY_avg = np.roll(negpseudoCyclesY_avg, -idx)

fig.add_trace(
  go.Scatter(
    name="Average normalized pseudo-cycle (negative frontier)", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=negpseudoCyclesY_avg,
    # fill="toself",
    mode="lines",
    line=dict(
        width=3,
        color="silver",
        # dash='dash'
        # showscale=False
    ),
    # visible = "legendonly"
  ),row=1, col=2
)

# Xcos = np.linspace(np.pi, - np.pi, maxL)
# fig.add_trace(
#   go.Scatter(
#     name="Sinusoid", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=Xcos,
#     y=np.cos(np.linspace(np.pi, 3 * np.pi, maxL)) * np.average(np.abs(negpseudoCyclesY)),
#     # fill="toself",
#     mode="lines",
#     line=dict(
#         width=4,
#         color="blue",
#         # showscale=False
#     ),
#     visible = "legendonly"
#   )
# )

fig.show(config=dict({'scrollZoom': True}))
fig.write_image(save_path, width=680, height=300, scale=1, engine="kaleido", format="svg")