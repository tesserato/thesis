import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import hp



'''==============='''
''' Read wav file '''
'''==============='''

name = "sin"

W, fps = se.read_wav(f"Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

Xpos_orig, Xneg_orig = se.get_frontiers(W)

Xpos = hp.refine_frontier_iter(Xpos_orig, W)
# while not Xposnew is None:
#   Xpos = np.unique(np.hstack([Xpos, Xposnew])).astype(np.int)
#   Xposnew = hp.refine_frontier(Xpos, W)

Xneg = hp.refine_frontier_iter(Xneg_orig, W)
# while not Xnegnew is None:
  # Xneg = np.unique(np.hstack([Xneg, Xnegnew])).astype(np.int)
  # Xnegnew = hp.refine_frontier(Xneg, W)

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
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=2, zerolinecolor='gray')


fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Refined Positive Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xpos,
    y=W[Xpos],
    # fill="toself",
    mode="markers",
    marker_symbol="diamond",
    marker=dict(size=4, color="gray"),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Refined Negative Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xneg,
    y=W[Xneg],
    # fill="toself",
    mode="markers",
    marker_symbol="square",
    marker=dict(size=4, color="gray"),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Positive Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xpos_orig,
    y=W[Xpos_orig],
    # fill="toself",
    mode="markers",
    marker_symbol="diamond",
    marker=dict(size=4, color="black"),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Negative Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xneg_orig,
    y=W[Xneg_orig],
    # fill="toself",
    mode="markers",
    marker_symbol="square",
    marker=dict(size=4, color="black"),
    # visible = "legendonly"
  )
)

fig.show()
# save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
# fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
# print("saved:", save_name)