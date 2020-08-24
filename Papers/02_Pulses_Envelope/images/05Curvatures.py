import sys
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
import os
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate.interpolate import interp1d
from Helper import read_wav, get_pulses_area, split_pulses, signal_to_pulses, get_frontier, draw_circle
from scipy.signal import savgol_filter



'''==============='''
''' Read wav file '''
'''==============='''
fig = go.Figure()

parent_path = str(Path(os.path.abspath('./')).parents[1])
print(parent_path)
name = "piano33"
W, fps = read_wav(f"{parent_path}/Python/Samples/{name}.wav")

W = W[100103 : 104200] #* 1000 / np.max(np.abs(W[100103 : 100775]))

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
pos_frontier_Y = pos_frontier_Y / scaling

'''neg frontier'''
scaling = np.average(neg_L) / np.average(neg_Y)

for p in neg_pulses:
  p.y = p.y * scaling

neg_frontier_X, neg_frontier_Y = get_frontier(neg_pulses)
neg_frontier_Y = np.array(neg_frontier_Y) / scaling

f_pos = interp1d(pos_frontier_X, pos_frontier_Y, fill_value="extrapolate", kind="linear")
f_neg = interp1d(neg_frontier_X, neg_frontier_Y, fill_value="extrapolate", kind="linear")
XX = np.linspace(0, n, 4 * n)
envelope = (f_pos(XX) - f_neg(XX)) / 2
envelope = savgol_filter(envelope, 101, 3)


'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''


fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="$i$",
  yaxis_title="$i$",
  # font_family="Courier New",
  yaxis = dict(scaleanchor = "x", scaleratio = 1), # <|<|<|<|<|<|<|<|<|<|<|<|
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
    family="Latin Modern",
    color="black",
    size=18
  )
)
fig.update_xaxes(showline=False, showgrid=False, zeroline=False, range=[0, 2000])
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver', range=[0, 800])

fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
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

fig.add_trace(
  go.Scatter(
    name="-Amps", # <|<|<|<|<|<|<|<|<|<|<|<|
    showlegend=False,
    x=neg_X,
    y=neg_Y,
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

# m0 = np.average(Y[1:] - Y[:-1]) / np.average(X[1:] - X[:-1])
# curvatures_X = []
# curvatures_Y = []
# for i in range(len(X) - 1):
#   m1 = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])

#   theta = np.arctan( (m1 - m0) / (1 + m1 * m0) )
#   k = np.sin(theta) / (X[i + 1] - X[i])

'''Positive Vectors'''

avg_pos_X = np.average(pos_X[1:] - pos_X[:-1])
avg_pos_Y = np.average(pos_Y[1:] - pos_Y[:-1])
avg_pos_m = avg_pos_Y / avg_pos_X #np.sqrt(avg_pos_X**2 + avg_pos_Y**2)

for i in range(len(pos_X) - 1):
  x0, y0 = pos_X[i]     , pos_Y[i]
  x1, y1 = pos_X[i + 1] , pos_Y[i + 1]

  # m = (y1 - y0) / (x1 - x0)
  # theta = np.arctan( (m - avg_pos_m) / (1 + avg_pos_m * m) )
  # k = np.sin(theta) / (x1 - x0)
  # r = 2 / np.abs(k)
  # sq = np.sqrt(4*m**2*r**2 - m**2*y0**2 + 2*m**2*y0*y1 - m**2*y1**2 - 2*m*x0*y0 + 2*m*x0*y1 + 2*m*x1*y0 - 2*m*x1*y1 + 4*r**2 - x0**2 + 2*x0*x1 - x1**2)
  # xc = (2*m**2*x0 - m*y0 + m*y1 - m*sq + x0 + x1)/(2*(m**2 + 1))
  # yc = (m*(y0 + y1) - 2*xc + x0 + x1)/(2*m)

  # xc = (avg_pos_m*x0*x1 - avg_pos_m*x1**2 - avg_pos_m*y0**2 + 2*avg_pos_m*y0*y1 - avg_pos_m*y1**2 - x0*y0 + x0*y1)/(avg_pos_m*x0 - avg_pos_m*x1 - y0 + y1)
  # yc = (avg_pos_m*y0 - xc + x0)/avg_pos_m
  # r = np.sqrt((xc - x0)**2 + (yc - y0)**2)

  # m = (y1 - y0) / (x1 - x0)
  # theta = np.arctan( (m - avg_pos_m) / (1 + avg_pos_m * m) )
  # c = np.sqrt((y1 - y0)**2 + (x1 - x0)**2)
  # r = c / (2 * np.sin((np.pi - np.arctan(theta)) / 2))
  # sq = np.sqrt(4*m**2*r**2 - m**2*y0**2 + 2*m**2*y0*y1 - m**2*y1**2 - 2*m*x0*y0 + 2*m*x0*y1 + 2*m*x1*y0 - 2*m*x1*y1 + 4*r**2 - x0**2 + 2*x0*x1 - x1**2)
  # xc = (2*m**2*x0 - m*y0 + m*y1 - m*sq + x0 + x1)/(2*(m**2 + 1))
  # yc = (m*(y0 + y1) - 2*xc + x0 + x1)/(2*m)

  m = (y1 - y0) / (x1 - x0)
  theta = np.arctan( (m - avg_pos_m) / (1 + avg_pos_m * m) )
  
  r = (np.sqrt(avg_pos_X**2 + avg_pos_Y**2) + np.sqrt((y1 - y0)**2 + (x1 - x0)**2)) / theta
  sq = np.sqrt(4*m**2*r**2 - m**2*y0**2 + 2*m**2*y0*y1 - m**2*y1**2 - 2*m*x0*y0 + 2*m*x0*y1 + 2*m*x1*y0 - 2*m*x1*y1 + 4*r**2 - x0**2 + 2*x0*x1 - x1**2)
  xc = (2*m**2*x0 - m*y0 + m*y1 - m*sq + x0 + x1)/(2*(m**2 + 1))
  yc = (m*(y0 + y1) - 2*xc + x0 + x1)/(2*m)

  draw_circle(xc, yc, r, fig)

  fig.add_annotation(
    # xanchor='right',
    yanchor='bottom',
    x=x1,  # arrows' head
    y=y1,  # arrows' head
    ax=x0,  # arrows' tail
    ay=y0,  # arrows' tail
    xref='x',
    yref='y',
    axref='x',
    ayref='y',
    text="$\\vec{v}["+ f"{i}]^+$",  # if you want only the arrow
    font=dict(size=14, color="black"),
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor='gray'
  )

  # fig.add_annotation(
  #   x=x0,  # arrows' head
  #   y=y0,  # arrows' head
  #   ax=x0 - avg_pos_X,  # arrows' tail
  #   ay=y0 - avg_pos_Y,  # arrows' tail
  #   xref='x',
  #   yref='y',
  #   axref='x',
  #   ayref='y',
  #   text='',  # if you want only the arrow
  #   showarrow=True,
  #   arrowhead=2,
  #   arrowsize=1,
  #   arrowwidth=2,
  #   arrowcolor='gray'
  # )

fig.add_annotation(
  x=avg_pos_X,        # arrows' head
  y=400 + avg_pos_Y,  # arrows' head
  ax=0,               # arrows' tail
  ay=400,             # arrows' tail
  xref='x',
  yref='y',
  axref='x',
  ayref='y',
  text='$\\vec{\\overline{v}}^{+}$',            # if you want only the arrow
  showarrow=True,
  arrowhead=2,
  arrowsize=1,
  arrowwidth=2,
  arrowcolor='gray'
)

# avg_neg_X = np.average(np.array(neg_frontier_X[1:]) - np.array(neg_frontier_X[:-1]))
# avg_neg_Y = np.average(np.array(neg_frontier_Y[1:]) - np.array(neg_frontier_Y[:-1]))

# for i in range(len(neg_frontier_X) - 1):
#   x0, y0 = neg_frontier_X[i], neg_frontier_Y[i]
#   x1, y1 = neg_frontier_X[i + 1], neg_frontier_Y[i + 1]
#   fig.add_annotation(
#     x=x1,  # arrows' head
#     y=y1,  # arrows' head
#     ax=x0,  # arrows' tail
#     ay=y0,  # arrows' tail
#     xref='x',
#     yref='y',
#     axref='x',
#     ayref='y',
#     text='',  # if you want only the arrow
#     showarrow=True,
#     arrowhead=2,
#     arrowsize=1,
#     arrowwidth=2,
#     arrowcolor='black'
#   )

#   fig.add_annotation(
#     x=x0 + avg_neg_X,  # arrows' head
#     y=y0 + avg_neg_Y,  # arrows' head
#     ax=x0,  # arrows' tail
#     ay=y0,  # arrows' tail
#     xref='x',
#     yref='y',
#     axref='x',
#     ayref='y',
#     text='',  # if you want only the arrow
#     showarrow=True,
#     arrowhead=2,
#     arrowsize=1,
#     arrowwidth=2,
#     arrowcolor='gray'
#   )

# fig.add_annotation(
#   x=avg_neg_X,        # arrows' head
#   y=-0.25 + avg_neg_Y,  # arrows' head
#   ax=0,               # arrows' tail
#   ay=-0.25,             # arrows' tail
#   xref='x',
#   yref='y',
#   axref='x',
#   ayref='y',
#   text='$\\vec{\\overline{v}}^{-}$',            # if you want only the arrow
#   showarrow=True,
#   arrowhead=2,
#   arrowsize=1,
#   arrowwidth=2,
#   arrowcolor='gray'
# )

# fig.show(config=dict({'scrollZoom': True}))
save_name = "./" + sys.argv[0].split('/')[-1].replace(".py", ".pdf")
fig.write_image(save_name, width=800, height=400, scale=1, engine="kaleido")
print("saved:", save_name)