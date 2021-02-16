from ctypes import resize
import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import sys
import os
os.chdir('../..')
print (os.path.abspath(os.curdir))

'''==============='''
''' Read wav file '''
'''==============='''

name = "piano33"
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
fig = go.Figure()
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


pospseudoCyclesY_avg = np.average(pospseudoCyclesY, 0)
fig.add_trace(
  go.Scatter(
    name="Average normalized pseudo-cycle (positive frontier)", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=pospseudoCyclesY_avg,
    # fill="toself",
    mode="lines",
    line=dict(
        width=3,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

negpseudoCyclesY_avg = np.average(negpseudoCyclesY, 0)
idx = np.argmax(negpseudoCyclesY_avg)
negpseudoCyclesY_avg = np.roll(negpseudoCyclesY_avg, -idx)

fig.add_trace(
  go.Scatter(
    name="Average normalized pseudo-cycle (negative frontier)", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pseudoCyclesX for item in sublist.tolist() + [None]],
    y=negpseudoCyclesY_avg,
    # fill="toself",
    mode="lines",
    line=dict(
        width=3,
        color="black",
        dash='dash'
        # showscale=False
    ),
    # visible = "legendonly"
  )
)


M = np.max(pospseudoCyclesY_avg)
fig.add_trace(
  go.Scatter(
    name="Average normalized pseudo-cycle (negative frontier)", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, negpseudoCyclesY_avg.size - 1],
    y=[M, M],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="gray",
        # dash='dash'
        # showscale=False
    ),
    showlegend=False
    # visible = "legendonly"
  )
)

M = np.max(negpseudoCyclesY_avg)
fig.add_trace(
  go.Scatter(
    name="Average normalized pseudo-cycle (negative frontier)", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, negpseudoCyclesY_avg.size - 1],
    y=[M, M],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        dash='dash'
        # showscale=False
    ),
    showlegend=False
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))
fig.write_image(save_path, width=680, height=300, scale=1, engine="kaleido", format="svg")