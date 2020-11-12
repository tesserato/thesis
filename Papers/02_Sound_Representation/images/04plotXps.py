import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
import sys
sys.path.insert(0,"C:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
import hp as hp


'''==============='''
''' Read wav file '''
'''==============='''


name = "piano33"
W, fps = se.read_wav(f"C:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos_orig, Xneg_orig = se.get_frontiers(W)

Xpos = hp.refine_frontier_iter(Xpos_orig, W)
Xneg = hp.refine_frontier_iter(Xneg_orig, W)



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
  xaxis_title="<b>Ordinality</b>",
  yaxis_title="<b><i>i</i></b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT
fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver')
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver')

fig.add_trace(
  go.Scatter(
    name="Maxima", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xpos_orig,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="black",
        dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Minima", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xneg_orig,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="gray",
        dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Refined Maxima", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xpos,
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

fig.add_trace(
  go.Scatter(
    name="Refined Minima", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=Xpos,
    y=Xneg,
    # fill="toself",
    mode="lines",
    line=dict(
        width=2,
        color="gray",
        # dash="dash"
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
print("saved:", save_name)