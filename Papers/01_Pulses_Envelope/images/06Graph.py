import sys
# sys.path.append("signal_envelope/")
import numpy as np
import signal_envelope as se
from scipy import interpolate
import plotly.graph_objects as go

# name = "alto"
# name="bend"
name="spoken_voice"
W, _ = se.read_wav(f"test_samples/{name}.wav")
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
fig.update_yaxes(showline=False, zeroline=False, showgrid=False, gridwidth=1, gridcolor='gray', tickvals=[-1, 0, 1])


fig.add_trace(
  go.Scatter(
    name="Signal (<b>w = e</b> ⊙ <b>c</b>)      ",
    # x=np.arange(W.size),
    y=W,
    mode="lines",
    line_shape='spline',
    # fillcolor="gray",
    line=dict(width=1, color="silver",),
  )
)

fig.add_trace(
  go.Scatter(
    name="Carrier (<b>c</b>)      ",
    # x=np.arange(W.size),
    y=C,
    mode="lines",
    line_shape='spline',
    # fillcolor="rgba(0,0,0,0.3)",
    line=dict(width=1, color="rgba(235, 64, 52,0.3)",),
  )
)

fig.add_trace(
  go.Scatter(
    name="Envelope (<b>e</b>)      ",
    line_shape='spline',
    y=E,
    mode="lines",
    line=dict(width=2, color="black"),
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Positive Frontier",
#     x=Xpos,
#     y=W[Xpos],
#     mode="lines",
#     line=dict(width=1, color="red"),
#     visible = "legendonly"
#   )
# )

# fig.add_trace(
#   go.Scatter(
#     name="Negative Frontier",
#     x=Xneg,
#     y=W[Xneg],
#     mode="lines",
#     line=dict(width=1, color="red"),
#     visible = "legendonly"
#   )
# )

# fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", "") + "_" + name + ".svg"
fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
print("saved:", save_name)


