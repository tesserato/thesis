import sys
print(sys.version)
import plotly.graph_objects as go
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/01_Envelope")
import numpy as np
from scipy.spatial import ConvexHull
from Helper import signal_to_pulses, get_pulses_area, alpha_shape, get_circle


def draw_circle(xc, yc, r, fig, n=100, color="silver"):
  '''draws circle as plotly scatter from center and radius'''
  X = []
  Y = []
  pulses_color = "rgba(250, 140, 132, 0.5)"

  for t in np.linspace(0, 2 * np.pi, n):
    X.append(xc + r * np.cos(t))
    Y.append(yc + r * np.sin(t))


  fig.add_trace(
    go.Scatter(
      name=f"Circle of radius \u03B1={radius}",
      # legendgroup="Circles",
      x=X,
      y=Y,
      # showlegend=False,
      # visible = "legendonly",
      mode="lines",
      line=dict(
          width=1,
          color="rgba(3, 19, 252, 0.8)"
      )
    )
  )

  fig.add_trace(
    go.Scatter(
      name="Open disc",
      # legendgroup="Circles",
      x=X,
      y=Y,
      fill="tozeroy",
      fillcolor="rgba(3, 19, 252, 0.1)",
      mode="none",
      # line=dict(
      #     width=1,
      #     color=color
      # )
    )
  )
  return X, Y



np.random.seed(1)
# fps = 10
n = 50+1

X = np.arange(n)# / fps

W = np.random.normal(0, np.sqrt(n), n)

W -= W.min()

radius = 10



points = np.array([[x, w] for x, w in zip(X, W)])
hull = ConvexHull(points)

x0, y0 = get_circle(points[10][0], points[10][1], points[14][0], points[14][1], radius)

'''============================================================================'''
'''                              Alpha 1                                    '''
'''============================================================================'''


alpha_edges = alpha_shape(points, radius)
alpha_edges = sorted(alpha_edges, key=lambda tup: tup[0])

XX1 = []
YY1 = []
for e in alpha_edges:
  idx0 = e[0]
  idx1 = e[1]
  XX1.append(points[idx0][0]), XX1.append(points[idx1][0]), XX1.append(None)
  YY1.append(points[idx0][1]), YY1.append(points[idx1][1]), YY1.append(None)


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
  xaxis = dict(scaleanchor = "y", scaleratio = 1),
  xaxis_title="<b><i>x</i></b>",
  yaxis_title="<b><i>y</i></b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1, itemsizing='trace'),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False, range=[0, 60])#, zerolinewidth=1, zerolinecolor='black')
fig.update_yaxes(showline=False, showgrid=False, zeroline=False)#, zerolinewidth=1, zerolinecolor='black')


fig.add_trace(
  go.Scatter(
    name="Set of points",
    # showlegend=False,
    x=X,
    y=W,
    mode='markers',
    line=dict(color="silver", width=.5),
    marker=dict(size=4, color="black")
  )
)

fig.add_trace(
  go.Scatter(
    name=f"Alpha shape (\u03B1={radius})    ",
    # showlegend=False,
    x=XX1,
    y=YY1,
    # fill="tozeroy",
    # fillcolor="rgba(0,0,0,0.16)",
    mode="lines",
    line=dict(color=red, width=2,),
    # line=dict(color="black", width=1),
    # marker=dict(size=3, color="black")
  )
)

draw_circle(x0, y0, radius, fig)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=400, engine="kaleido", format="svg")
print("saved:", save_name)
