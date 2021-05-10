import sys
print(sys.version)
import plotly.graph_objects as go
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/01_Envelope")
import numpy as np
from scipy.interpolate import interp1d
from Helper import signal_to_pulses, get_pulses_area
import signal_envelope as se

name = "tom"

'''==============='''
''' Read wav file '''
'''==============='''
W, fps = se.read_wav(f"test_samples/{name}.wav")
W = W[13141 : 17281]
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
print(f"n={n}")
X = np.arange(n)


t = int(n / np.argmax(np.abs(np.fft.rfft(W))))
print(t)
rolled_W = np.roll(np.pad(W, (t//2, t//2) ), -t//2)

Xf = []
Yf = []
for i in range(t):
  rolled_W = np.roll(rolled_W, 1)
  Yf.append(rolled_W)
  Xf.append(np.arange(rolled_W.size) - t//2)

Xf = np.array(Xf)
Yf = np.array(Yf)




'''============================================================================'''
'''                              PLOT LINES                                    '''
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
  xaxis_title="<b>Sample <i>i</i></b>",
  yaxis_title="<b>Amplitude</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1, itemsizing='constant'),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, zeroline=False, showgrid=True)

fig.add_trace(
  go.Scatter(
    name="Signal",
    # x=np.arange(rolled_W.size).tolist() + [None],
    y=W,
    mode="lines",
    line=dict(width=2, color="black"
    , shape = 'spline'
    ),
    marker=dict(size=3, color="red")
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="$ \\text{Family of signals, shifted from } i-T/2 \\text{ to } i + T/2 $",
    x=Xf[0:-1:20].flatten()[t : ],#pulses_X,
    y=Yf[0:-1:20].flatten()[t : ],#pulses_Y,
    mode="lines",
    line=dict(width=1, color="rgba(120, 120, 120, .5)"
    # , shape = 'spline'
    ),
    marker=dict(size=4, color="gray"),
    # showlegend=False
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Reference envelope",
    x=(np.arange(rolled_W.size) - t//2)[t//2 : ],
    y=(np.max(np.array(Yf)[:, 0:-1], 0)[t//2 : ]),
    mode="lines",
    line=dict(width=2, color="red"
    # , shape = 'spline'
    ),
    marker=dict(size=3, color="red")
    # visible = "legendonly"
  )
)

# fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=350, engine="kaleido", format="svg")
print("saved:", save_name)
