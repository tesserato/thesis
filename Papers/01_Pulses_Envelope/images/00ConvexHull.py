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

'''============================================================================'''
'''                              Alpha 1                                    '''
'''============================================================================'''

alpha1 = 50
alpha_edges = alpha_shape(points, alpha1)
alpha_edges = sorted(alpha_edges, key=lambda tup: tup[0])

XX1 = []
YY1 = []
for e in alpha_edges:
  idx0 = e[0]
  idx1 = e[1]
  XX1.append(points[idx0][0]), XX1.append(points[idx1][0]), XX1.append(None)
  YY1.append(points[idx0][1]), YY1.append(points[idx1][1]), YY1.append(None)

'''============================================================================'''
'''                              Alpha 2                                    '''
'''============================================================================'''

alpha2 = 300
alpha_edges = alpha_shape(points, alpha2)
alpha_edges = sorted(alpha_edges, key=lambda tup: tup[0])

XX2 = []
YY2 = []
for e in alpha_edges:
  idx0 = e[0]
  idx1 = e[1]
  XX2.append(points[idx0][0]), XX2.append(points[idx1][0]), XX2.append(None)
  YY2.append(points[idx0][1]), YY2.append(points[idx1][1]), YY2.append(None)


'''============================================================================'''
'''                              PLOT LINES                                    '''
'''============================================================================'''
FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

al = .5
red = f"rgba(191, 25, 13, {al})"
blue = f"rgba(13, 92, 161, {al})"
green = f"rgba(0, 0, 0, {al})"

'''Plotting'''
fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="<b><i>x</i></b>",
  yaxis_title="<b><i>y</i></b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1, itemsizing='trace'),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')



fig.add_trace(
  go.Scatter(
    name=f"Alpha-Shape (\u03B1={alpha1})    ",
    # showlegend=False,
    x=XX1,
    y=YY1,
    # fill="tozeroy",
    fillcolor="rgba(0,0,0,0.16)",
    mode="lines",
    line=dict(color=red, width=2, dash="dash"),
    # line=dict(color="black", width=1),
    # marker=dict(size=3, color="black")
  )
)

fig.add_trace(
  go.Scatter(
    name=f"Alpha-Shape (\u03B1={alpha2})    ",
    # showlegend=False,
    x=XX2,
    y=YY2,
    # fill="tozeroy",
    # fillcolor=red,
    mode="lines",
    line=dict(color=blue, width=2, dash="dot"),
    # line=dict(color="black", width=1),
    # marker=dict(size=3, color="black")
  )
)

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
    line=dict(color=green, width=2),
    # marker=dict(size=3, color="black")
    # visible = "legendonly"
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
    marker=dict(size=4, color="black")
  )
)

# fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
print("saved:", save_name)
