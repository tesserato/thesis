import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import sys

def refine_frontier(Xp, W):
  "find additional frontier points, and return an array with then, if found"
  if W[Xp[0]] >= 0:
    e = np.argmax
  else:
    e = np.argmin
  L = Xp[1:] - Xp[:-1]
  avgL = np.average(L)
  stdL = np.std(L)
  Xnew = []
  for i in range(1, Xp.size):
    x0 = int(Xp[i - 1])
    x1 = int(Xp[i])
    if x1 - x0 > avgL + 2 * stdL:
      Xzeroes = []
      currsign = np.sign(W[x0])
      for i in range(x0 + 1, x1):
        if currsign != np.sign(W[i]):
          Xzeroes.append(i)
          currsign = np.sign(W[i])
      if len(Xzeroes) > 1:
        Xnew.append(Xzeroes[0] + e(W[Xzeroes[0] : Xzeroes[-1]]))
  return np.array(Xnew, dtype=np.int)


'''==============='''
''' Read wav file '''
'''==============='''

name = "tom"

W, fps = se.read_wav(f"C:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)

Xposnew = refine_frontier(Xpos, W)
Xnegnew = refine_frontier(Xneg, W)



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
fig.update_xaxes(showline=False, showgrid=False, zeroline=False, range=[29872, 33704])
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=2, range=[-4100, 4000], zerolinecolor='gray')


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
    name="Positive Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xpos,
    y=W[Xpos],
    # fill="toself",
    mode="markers",
    marker_symbol="diamond",
    marker=dict(size=10, color="black"),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Negative Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xneg,
    y=W[Xneg],
    # fill="toself",
    mode="markers",
    marker_symbol="square",
    marker=dict(size=10, color="black"),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Refined Positive Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xposnew,
    y=W[Xposnew],
    # fill="toself",
    mode="markers",
    marker_symbol="diamond",
    marker=dict(size=10, color="gray"),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Refined Negative Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xnegnew,
    y=W[Xnegnew],
    # fill="toself",
    mode="markers",
    marker_symbol="square",
    marker=dict(size=10, color="gray"),
    # visible = "legendonly"
  )
)

fig.show()
# save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
# fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
# print("saved:", save_name)