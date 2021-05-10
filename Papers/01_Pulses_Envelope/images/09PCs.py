'''To be used AFTER signal_envelope is installed via pip'''
import sys
# sys.path.append("signal_envelope/")
import numpy as np
import signal_envelope as se
from scipy import interpolate
import plotly.graph_objects as go

name = "alto"
W, fps = se.read_wav(f"test_samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
X = np.arange(W.size)

Xpos, Xneg = se.get_frontiers(W, 0)
E = se.get_frontiers(W, 1)

Xp = []
Yp = []
for x in Xpos:
  Xp.append(x), Xp.append(x), Xp.append(None)
  Yp.append(-1), Yp.append(1), Yp.append(None)


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
  yaxis_title="<b>Amplitude</b>",
  xaxis_title="<b>Sample <i>i</i></b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False, range=[21474, 22343])
fig.update_yaxes(showline=True, showgrid=False, zeroline=True, zerolinecolor="black", range=[-.75, 1])

fig.add_trace(
  go.Scatter(
    name="Signal      ",
    # x=E,
    y=W,
    mode="lines+markers",
    line=dict(width=1, color="black"),
    marker=dict(size=3),
  )
)

fig.add_trace(
  go.Scatter(
    name="Indices of the positive frontier      ",
    x=Xp,
    y=Yp,
    mode="lines",
    # fill="tozeroy",
    # fillcolor="rgba(0,0,0,0.6)",
    line=dict(width=1, color="blue",),
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Negative Frontier      ",
#     # x=np.arange(W.size),
#     y=Xneg,
#     mode="lines+markers",
#     # fill="tozeroy",
#     # fillcolor="rgba(0,0,0,0.3)",
#     line=dict(width=1, color="red",),
#   )
# )



# fig.add_trace(
#   go.Scatter(
#     name="Envelope      ",
#     # x=E,
#     y=E,
#     mode="lines+markers",
#     line=dict(width=1, color="black"),
#   )
# )

# fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", "") + ".svg"
fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
print("saved:", save_name)
