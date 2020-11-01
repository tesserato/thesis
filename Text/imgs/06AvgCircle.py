import sys
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
import os
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate.interpolate import interp1d
from Helper import read_wav, get_pulses_area, split_pulses, signal_to_pulses, get_frontier, draw_circle, get_curvature_function
from scipy.signal import savgol_filter

# os.path.dirname(os.path.dirname(path))

'''==============='''
''' Read wav file '''
'''==============='''
fig = go.Figure()

parent_path = str(Path(os.path.abspath('./')).parents[0])
print(parent_path)
name = "piano33"
W, fps = read_wav(f"{parent_path}/Python/Samples/{name}.wav")

W = W[100103 : 100103 + 585] #* 1000 / np.max(np.abs(W[100103 : 100775]))

W = W - np.average(W)
# amplitude = np.max(np.abs(W))
W = W * .5
n = W.size
print(n)
X = np.arange(n)

pulses = signal_to_pulses(W)

areas_X, areas_Y = get_pulses_area(pulses)

pos_pulses, neg_pulses, pos_noises, neg_noises = split_pulses(pulses)

pos_X, neg_X = np.array([p.x for p in pos_pulses]), np.array([p.x for p in neg_pulses])
pos_Y, neg_Y = np.array([p.y for p in pos_pulses]), np.array([p.y for p in neg_pulses])
pos_L, neg_L = np.array([p.len for p in pos_pulses]) , np.array([p.len for p in neg_pulses])

'''pos frontier'''
scaling = np.average(pos_L) / np.average(pos_Y)

for p in pos_pulses:
  p.y = p.y * scaling

pos_frontier_X, pos_frontier_Y = get_frontier(pos_pulses)

pos_frontier_X = np.array(pos_frontier_X)
pos_frontier_Y = np.array(pos_frontier_Y)

pos_frontier_Y = pos_frontier_Y / scaling

'''neg frontier'''
scaling = np.average(neg_L) / np.average(neg_Y)

for p in neg_pulses:
  p.y = p.y * scaling

neg_frontier_X, neg_frontier_Y = get_frontier(neg_pulses)
neg_frontier_Y = np.array(neg_frontier_Y) / scaling

# f_pos = interp1d(pos_frontier_X, pos_frontier_Y, fill_value="extrapolate", kind="linear")
# f_neg = interp1d(neg_frontier_X, neg_frontier_Y, fill_value="extrapolate", kind="linear")
# XX = np.linspace(0, n, 4 * n)
# envelope = (f_pos(XX) - f_neg(XX)) / 2
# envelope = savgol_filter(envelope, 101, 3)


'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''

FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="<b><i>i</i></b>",
  yaxis_title="<b><i>i</i></b>",
  yaxis = dict(scaleanchor = "x", scaleratio = 1), # <|<|<|<|<|<|<|<|<|<|<|<|
  paper_bgcolor='rgba(0,0,0,0)',
  plot_bgcolor='rgba(0,0,0,0)',
  legend=dict(orientation='h', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT
fig.update_xaxes(showline=False, showgrid=False, zeroline=False, range=[0, 600])
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='gray', range=[0, 900])

fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="markers+lines",
    line=dict(
        width=.8,
        color="gray",
        # showscale=False
    ),
    marker=dict(
      size=3,
      color="gray",
      # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Pulses", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=areas_X,
    y=areas_Y,
    fill="tozeroy",
    mode="none",
    fillcolor="rgba(0,0,0,0.16)",
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+Amps", # <|<|<|<|<|<|<|<|<|<|<|<|
    showlegend=False,
    x=pos_X,
    y=pos_Y,
    # hovertext=np.arange(len(pos_pulses)),
    mode="markers",
    marker=dict(
        size=6,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="-Amps", # <|<|<|<|<|<|<|<|<|<|<|<|
#     showlegend=False,
#     x=neg_X,
#     y=neg_Y,
#     # hovertext=np.arange(len(pos_pulses)),
#     mode="markers",
#     marker=dict(
#         size=6,
#         color="black",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )



args = get_curvature_function(pos_X, pos_Y)
r = args[0](0)

'''Positive Vectors'''
avg_pos_X = np.average(pos_frontier_X[1:] - pos_frontier_X[:-1])
avg_pos_Y = np.average(pos_frontier_Y[1:] - pos_frontier_Y[:-1])
avg_pos_m = avg_pos_Y / avg_pos_X #np.sqrt(avg_pos_X**2 + avg_pos_Y**2)

for i in range(len(pos_frontier_X) - 1):
  x0, y0 = pos_frontier_X[i]     , pos_frontier_Y[i]
  x1, y1 = pos_frontier_X[i + 1] , pos_frontier_Y[i + 1]

  m = (y1 - y0) / (x1 - x0)
  # theta = np.arctan( (m - avg_pos_m) / (1 + avg_pos_m * m) )
  # k = np.sin(theta) / (x1 - x0)
  # r = 2 / np.abs(k)
  sq = np.sqrt(4*m**2*r**2 - m**2*y0**2 + 2*m**2*y0*y1 - m**2*y1**2 - 2*m*x0*y0 + 2*m*x0*y1 + 2*m*x1*y0 - 2*m*x1*y1 + 4*r**2 - x0**2 + 2*x0*x1 - x1**2)
  xc = (2*m**2*x0 - m*y0 + m*y1 - m*sq + x0 + x1)/(2*(m**2 + 1))
  yc = (m*(y0 + y1) - 2*xc + x0 + x1)/(2*m)

  draw_circle(xc, yc, r, fig, 1000)

  # fig.add_annotation(
  #   x=x1,  # arrows' head
  #   y=y1,  # arrows' head
  #   ax=x0,  # arrows' tail
  #   ay=y0,  # arrows' tail
  #   xref='x',
  #   yref='y',
  #   axref='x',
  #   ayref='y',
  #   text='',  # if you want only the arrow
  #   showarrow=True,
  #   arrowhead=2,
  #   arrowsize=1,
  #   arrowwidth=2,
  #   arrowcolor="black"
  # )

fig.add_trace(
  go.Scatter(
    name="Positive Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=pos_frontier_X,
    y=pos_frontier_Y,
    mode="lines",
    line=dict(
        width=2,
        color="black",
        # showscale=False
    ),
    # marker=dict(
    #   size=3,
    #   color="gray",
    #   # showscale=False
    # ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Circles", # <|<|<|<|<|<|<|<|<|<|<|<|
    # showlegend=False,
    x=[None],
    y=[None],
    # hovertext=np.arange(len(pos_pulses)),
    mode="lines",
    line=dict(
        width=1,
        color="silver",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

# fig.show()
save_name = "./imgs/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=605, height=400, engine="kaleido", format="svg")
print("saved:", save_name)
