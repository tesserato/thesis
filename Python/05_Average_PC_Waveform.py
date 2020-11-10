import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp
from statistics import mode
from scipy import interpolate

'''==============='''
''' Read wav file '''
'''==============='''

name = "piano33"
W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)

Xpos = hp.refine_frontier_iter(Xpos, W)
Xneg = hp.refine_frontier_iter(Xneg, W)

print("Sizes:", Xpos.size, Xneg.size)

Y = hp.average_pc_waveform(Xpos, Xneg, W)


'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",
  # yaxis = dict(scaleanchor = "x", scaleratio = 1 ),                   # <|<|<|<|<|<|<|<|<|<|<|<|
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

fig.add_trace(
  go.Scattergl(
    name="+Pseudo-Cycles", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=[item for sublist in pospseudoCyclesY for item in Xpc + [None]],
    y=Y,
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

fig.show(config=dict({'scrollZoom': True}))