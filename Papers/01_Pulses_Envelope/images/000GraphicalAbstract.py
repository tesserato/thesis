import sys
import numpy as np
import signal_envelope as se
from scipy import interpolate
import plotly.graph_objects as go
from plotly.subplots import make_subplots


name = "piano"
W, _ = se.read_wav(f"C:/Users/tesse/Desktop/Files/Dropbox/0_Science/reps/envelope/test_samples/{name}.wav")
amp = np.max(np.abs(W))
W = 4 * W / amp
X = np.arange(W.size)

Xpos, Xneg = se.get_frontiers(W, 0)
E = se.get_frontiers(W, 1)

f = interpolate.interp1d(E, np.abs(W[E]), kind="linear", fill_value="extrapolate")
E = f(X)

for i in range(E.size):
  if E[i] < 0.1:
    E[i] = 0.1

C = W / E

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''

FONT = dict(
    family="Arial",
    color="black",
    size=24
    # size=13.3333
  )

'''Plotting'''
fig = make_subplots(
  rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01,
  subplot_titles=("<b>Extracted Envelope and Carrier Wave</b> (33<sup>rd</sup> key of a grand piano)", "<b>Extracted Superior and Inferior Frontiers</b> (33<sup>rd</sup> key of a grand piano)")
)
fig.layout.template ="plotly_white" 
fig.update_layout(
  # xaxis_title="<b><i>i</i></b>",
  # yaxis_title="<b>Amplitude</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  # title=dict(text="<b>Teste</b>", font=FONT)
)
# fig.layout.xaxis.title.font=FONT
# fig.layout.yaxis.title.font=FONT

fig.update_xaxes(
  showline=False, 
  showgrid=False, 
  zeroline=False, 
  showticklabels=False, 
  # title=dict(text="<b><i>i</i></b>",font=FONT), 
  row=1, col=1
)

fig.update_xaxes(
  showline=False, 
  showgrid=False, 
  zeroline=False, 
  showticklabels=True, 
  title=dict(text="<b><i>i</i></b>",font=FONT), 
  row=2, col=1
)

fig.update_yaxes(
  showline=False,
  showgrid=False,
  zeroline=False,
  gridcolor='gray',
  tickvals=[ -3, -1, 1, 3],
  title=dict(text="<b>Amplitude</b>", font=FONT),
  row=1, col=1
)

fig.update_yaxes(
  showline=False, 
  showgrid=False, 
  zerolinewidth=1, 
  zerolinecolor="gray",
  showticklabels=False, 
  title=dict(text="<b>Amplitude</b>",font=FONT), 
  row=2, col=1
)

for i in fig['layout']['annotations']:
    i['font'] = FONT


fig.add_trace(
  go.Scatter(
    name="Original Signal   ",
    # x=np.arange(W.size),
    y=W,
    mode="lines",
    # fill="tozeroy",
    # fillcolor="gray",
    line=dict(width=1, color="rgba(0, 0, 0, 0.4)",),
  ), row=1, col=1
)

fig.add_trace(
  go.Scatter(
    name="Carrier   ",
    # x=np.arange(W.size),
    y=C,
    mode="lines",
    # fill="tozeroy",
    # fillcolor="rgba(0,0,0,0.3)",
    line=dict(width=1, color="rgba(235, 64, 52, 0.3)",),
  ), row=1, col=1
)

fig.add_trace(
  go.Scatter(
    name="Envelope   ",
    # x=E,
    y=E,
    mode="lines",
    line=dict(width=2, color="black"),
  ), row=1, col=1
)

'''Plot 2'''
fig.add_trace(
  go.Scatter(
    name="Original Signal      ",
    showlegend = False,
    # x=np.arange(W.size),
    y=W,
    mode="lines",
    # fill="tozeroy",
    # fillcolor="gray",
    line=dict(width=1, color="rgba(0, 0, 0, 0.4)",),
  ), row=2, col=1
)

fig.add_trace(
  go.Scatter(
    name="Superior Frontier   ",
    x=Xpos,
    y=W[Xpos],
    mode="lines",
    line=dict(width=2, color="blue"),
    # visible = "legendonly"
  ), row=2, col=1
)

fig.add_trace(
  go.Scatter(
    name="Inferior Frontier   ",
    x=Xneg,
    y=W[Xneg],
    mode="lines",
    line=dict(width=2, color="red"),
    # visible = "legendonly"
  ), row=2, col=1
)

fig.layout.annotations[0].update(y=0.85)
fig.layout.annotations[1].update(y=0.35)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", "") + ".svg"
fig.write_image(save_name, width=1328, height=531, engine="kaleido", format="svg")
print("saved:", save_name)


