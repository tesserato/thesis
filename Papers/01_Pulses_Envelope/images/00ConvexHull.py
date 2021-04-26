import sys
print(sys.version)
import plotly.graph_objects as go
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/01_Envelope")
import numpy as np
from scipy.spatial import ConvexHull
from Helper import signal_to_pulses, get_pulses_area, alpha_shape
# from plotly.subplots import make_subplots
'''Generating Wave'''

np.random.seed(1)
# fps = 10
n = 200+1

X = np.arange(n)# / fps
W = np.zeros(n)

afp = [
  [1, 19, np.pi],
  [1, 29, 1.2 * np.pi],
  [.6, 9.3, 1.4 * np.pi]
  ]
for a, f, p in afp:
  W += a * np.cos(p + 2 * np.pi * f * X / n) + np.random.normal(0, a / 20, n)

W = W / np.max(np.abs(W))



points = np.array([[x, w] for x, w in zip(X, W)])
hull = ConvexHull(points)
alpha = 300
alpha_edges = alpha_shape(points, alpha)
alpha_edges = sorted(alpha_edges, key=lambda tup: tup[0])

XX = []
YY = []
for e in alpha_edges:
  idx0 = e[0]
  idx1 = e[1]

  XX.append(points[idx0][0]), XX.append(points[idx1][0]), XX.append(None)
  YY.append(points[idx0][1]), YY.append(points[idx1][1]), YY.append(None)


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
  xaxis_title="<b><i>x</i></b>",
  yaxis_title="<b><i>y</i></b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')


I = [v for v in hull.vertices] + [hull.vertices[0]]
fig.add_trace(
  go.Scatter(
    name="Convex Hull",
    # showlegend=False,
    x=X[I],
    y=W[I],
    # fill="toself",
    # fillcolor="rgba(0,0,0,0.16)",
    mode="lines",
    line=dict(color="blue", width=4),
    # marker=dict(size=3, color="black")
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name=f"Concave Hull (Alpha-Shape, \u03B1={alpha})      ",
    # showlegend=False,
    x=XX,
    y=YY,
    # fill="tozeroy",
    fillcolor="rgba(0,0,0,0.16)",
    mode="lines",
    line=dict(color="red", width=2),
    # line=dict(color="black", width=1),
    # marker=dict(size=3, color="black")
  )
)

'''Samples'''
fig.add_trace(
  go.Scatter(
    name="Signal",
    # showlegend=False,
    x=X,
    y=W,
    mode='lines+markers',
    line=dict(color="silver", width=.5),
    marker=dict(size=5, color="black")
  )
)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
print("saved:", save_name)
