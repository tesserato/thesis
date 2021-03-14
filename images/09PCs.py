'''To be used AFTER signal_envelope is installed via pip'''
import sys
# sys.path.append("signal_envelope/")
import numpy as np
import signal_envelope as se
from scipy import interpolate
import plotly.graph_objects as go

name = "alto"
W, _ = se.read_wav(f"C:/Users/tesse/Desktop/Files/Dropbox/0_Science/reps/envelope/test_samples/{name}.wav")
amp = np.max(np.abs(W))
W = 4 * W / amp
X = np.arange(W.size)

Xpos, Xneg = se.get_frontiers(W, 0)
E = se.get_frontiers(W, 1)

m = min(Xpos.size, Xneg.size)
Xavg = (Xpos[0 : m] + Xneg[0 : m]) / 2

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
  xaxis_title="",
  yaxis_title="<b><i>i</i></b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False, range=[288-.5, 293+.5])
fig.update_yaxes(showline=False, showgrid=False, zeroline=False, range=[23500, 24030])


fig.add_trace(
  go.Scatter(
    name="Positive Frontier      ",
    # x=np.arange(W.size),
    y=Xpos,
    mode="lines+markers",
    # fill="tozeroy",
    # fillcolor="rgba(0,0,0,0.6)",
    line=dict(width=1, color="blue",),
  )
)

fig.add_trace(
  go.Scatter(
    name="Negative Frontier      ",
    # x=np.arange(W.size),
    y=Xneg,
    mode="lines+markers",
    # fill="tozeroy",
    # fillcolor="rgba(0,0,0,0.3)",
    line=dict(width=1, color="red",),
  )
)

fig.add_trace(
  go.Scatter(
    name="Average      ",
    # x=E,
    y=Xavg,
    mode="lines+markers",
    line=dict(width=1, color="gray"),
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="Envelope      ",
#     # x=E,
#     y=E,
#     mode="lines+markers",
#     line=dict(width=1, color="black"),
#   )
# )

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", "") + ".svg"
fig.write_image(save_name, width=650, height=200, engine="kaleido", format="svg")
print("saved:", save_name)
