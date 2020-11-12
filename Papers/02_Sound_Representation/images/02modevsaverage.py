import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
from statistics import mode
import sys
sys.path.insert(0,"C:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
import hp as hp

'''==============='''
''' Read wav file '''
'''==============='''

# def stats(L):
#   avg = np.average(L)
#   mde = mode(L)
#   std_avg = np.average(np.abs(L - avg))
#   std_mde = np.average(np.abs(L - mde))
#   print("std:", std_avg, std_mde, np.min(L), mde, np.max(L))
#   return avg, std_avg, mde, std_mde, np.std(L)

name = "piano33"
W, fps = se.read_wav(f"C:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/Samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
n = W.size
X = np.arange(n)

Xpos, Xneg = se.get_frontiers(W)
Xpos, Xneg = Xpos.astype(np.int), Xneg.astype(np.int)
posT = Xpos[1:] - Xpos[:-1]
negT = Xneg[1:] - Xneg[:-1]
T = np.hstack([posT, negT])

# Xpos = hp.refine_frontier_iter(Xpos, W)
# Xneg = hp.refine_frontier_iter(Xneg, W)
# posT_ref = Xpos[1:] - Xpos[:-1]
# negT_ref = Xneg[1:] - Xneg[:-1]
# T_ref = np.hstack([posT_ref, negT_ref])

mode_error = T - mode(T)
modeX, modeY = np.unique(mode_error, return_counts=True)

avg_error = T - np.average(T)
avgX, avgY = np.unique(avg_error, return_counts=True)

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
  xaxis_title="<b><i>T</i></b>",
  yaxis_title="<b>Number of Ocurrences</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT
fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver', range=[-5, 10])
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver')


fig.add_trace(
  go.Scatter(
    name=f"<b>Mode</b> (average absolute error={np.round(np.average(np.abs(mode_error)), 2)})", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=modeX,
    y=modeY,
    fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name=f"<b>Average</b> (average absolute error={np.round(np.average(np.abs(avg_error)), 2)})", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=avgX,
    y=avgY,
    fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
print("saved:", save_name)