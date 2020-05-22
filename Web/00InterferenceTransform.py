import numpy as np
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
r = np.array([[0, -1], [1, 0]])  # rotation matrix

class ParallelLines:
  Counter = 1
  Lines = []

  @classmethod
  def Plot(cls, ymax):
    for line in cls.Lines:
      line.plot(ymax=ymax)

  @classmethod
  def Intersection(cls):
    Sn  = np.zeros((2, 2))
    Snp = np.zeros((2, 1))
    for line in cls.Lines:
      nnT = line.n @ line.n.T
      Sn  += nnT * line.w
      Snp += nnT @ line.p * line.w
    return np.linalg.inv(Sn) @ Snp

  def __init__(self, t, n, weight=1):
    self.number = ParallelLines.Counter
    ParallelLines.Counter += 1
    ParallelLines.Lines.append(self)
    self.xv = (2 * np.pi * n**2) / (n**2 + 4 * np.pi**2 * t**2)
    self.yv = (4 * np.pi**2 * n * t) / (n**2 + 4 * np.pi**2 * t**2)
    self.t = t
    self.n = n
    self.w  = weight

    self.v = np.array([[self.xv], [self.yv]])    # vector perpendicular to line | point in line
    self.n = self.v / np.sqrt(self.v.T @ self.v) # unit vector perpendicular to line
    self.u = r @ self.n                          # unit vector in the direction of line

  def plot(self, xi=0, xf=2*np.pi, ymax=10):
    color = DEFAULT_PLOTLY_COLORS[(self.number - 1) % len(DEFAULT_PLOTLY_COLORS)]

    # plotting lines from x=xi to x=xf
    X = []
    Y = []
    k = 0
    while True:
      li = (xi - k * self.xv) / self.u[0, 0]
      lf = (xf - k * self.xv) / self.u[0, 0]
      pi = (k * self.v + li * self.u)
      pf = (k * self.v + lf * self.u)
      k += 1
      if max(pi[0, 0], pi[1, 0]) <= ymax:
        X.append(pi[0, 0]), X.append(pf[0, 0]), X.append(None)
        Y.append(pi[1, 0]), Y.append(pf[1, 0]), Y.append(None)
      else:
        break

    fig.add_trace(
      go.Scatter(
        name=f"t={self.t}",
        legendgroup=f"t={self.t}",
        x=X,
        y=Y,
        line = dict(color=color),
        mode="lines"
      ),
      row=1,
      col=2
    )
    # plotting V
    # line
    fig.add_trace(
      go.Scatter(
        name=f"V {self.number}",
        legendgroup=f"t={self.t}",
        showlegend=False,
        x=[0, self.xv],
        y=[0, self.yv],
        line = dict(color=color, dash='dash'),
        mode="lines"
      ),
      row=1,
      col=2
    )
    # plotting head
    fig.add_trace(
      go.Scatter(
        name=f"point {self.number}",
        legendgroup=f"t={self.t}",
        showlegend=False,
        x=[self.xv],
        y=[self.yv],
        mode='markers',
        marker=dict(color=color)
      ),
      row=1,
      col=2
    )
    # plotting N
    # line
    fig.add_trace(
      go.Scatter(
        name=f"V {self.number}",
        legendgroup=f"t={self.t}",
        showlegend=False,
        x=[0, self.n[0, 0]],
        y=[0, self.n[1, 0]],
        line = dict(color=color), mode="lines"
      ),
      row=1,
      col=2
    )
    # plotting head
    fig.add_trace(
      go.Scatter(
        name=f"PV {self.number}",
        legendgroup=f"t={self.t}",
        showlegend=False,
        x=[self.n[0, 0]],
        y=[self.n[1, 0]],
        mode='markers',
        marker=dict(color=color)
      ),
      row=1,
      col=2
    )
    # plotting U
    # line
    fig.add_trace(
      go.Scatter(
        name=f"U {self.number}",
        legendgroup=f"t={self.t}",
        showlegend=False,
        x=[0, self.u[0, 0]],
        y=[0, self.u[1, 0]],
        line = dict(color=color), mode="lines"
      ),
      row=1,
      col=2
    )
    # plotting head
    fig.add_trace(
      go.Scatter(
        name=f"PU {self.number}",
        legendgroup=f"t={self.t}",
        showlegend=False,
        x=[self.u[0, 0]],
        y=[self.u[1, 0]],
        mode='markers',
        marker=dict(color=color)
      ),
      row=1,
      col=2
    )


### Generating random wave
random.seed(1)
n = 10

X = np.arange(n)
W = np.zeros(n)

for _ in range(100):
  a = random.uniform(1, 5)
  f = random.uniform(1, 10)
  p = random.uniform(0, 2 * np.pi)
  W += a * np.cos(p + 2 * np.pi * f * X / n)

W = W / np.max(np.abs(W))

### Creating and formatting plotly figure
fig = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7])

fig.layout.template ="plotly_white"

fig.update_layout(
  xaxis_title="Frame",
  yaxis_title="Amplitude",
  yaxis = dict(scaleanchor = "x", scaleratio = 1),
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(family="Computer Modern", color="black", size=18),
  hoverlabel=dict(font_family="Computer Modern", font_size=18)
)

fig.update_xaxes(gridwidth=1, gridcolor='silver', tickvals=np.arange(n))
fig.update_yaxes(zerolinewidth=2, zerolinecolor='silver')

### plotting discrete wave
### stems
fig.add_trace(
  go.Scatter(
    name= "Discrete Wave Stem",
    showlegend=False,
    hoverinfo="none",
    x=[i for x in X for i in (x, x, None)],
    y=[i for y in W for i in (0, y, None)],
    mode='lines',
    line=go.scatter.Line(color="black", width=4)
  ),
  row=1,
  col=1
)
### markers
fig.add_trace(
  go.Scatter(
  name="",
  x=X,
  y=W,
  mode='markers',
  marker=dict(size=12, color="black"),
  hoverinfo="none",
  hovertemplate="t = %{x}<br>W[%{x}] = %{y:.2f}"
  ),
  row=1,
  col=1
)


for t in range(n):
  ParallelLines(t, n, W[t])
ParallelLines.Plot(ymax=n+1)

# fig.write_html("file.html", include_plotlyjs="cdn")
fig.show()
exit()

#######################
#######################
res_f = 200
res_p = 200
#######################
#######################

FP = np.zeros((res_f, res_p))
F = np.linspace(0, n, res_f)
P = np.linspace(0, 2 * np.pi, res_p)

X, Y = [], []
for t in range(n):
  for i in range(res_f):
    for j in range(res_p):
      FP[i, j] += W[t] * np.cos(P[j] + 2*np.pi*F[i]*t/n)
      X.append(P[j])
      Y.append(F[i])


fig = go.Figure(data=[go.Surface(x=P, y=F, z=FP)])
fig.layout.template ="plotly_dark"
fig.update_layout(
    font=dict(
      family="Computer Modern",
      color="white",
      size=16),
    scene_camera=dict(eye=dict(x=0, y=0, z=3), up=dict(x=0, y=-1, z=0),),
    scene = dict(
      xaxis_title="Phase",
      yaxis_title="Frequency",
      zaxis_title="Amplitude")
      )

fig.write_html("file.html", include_plotlyjs="cdn")

# app = dash.Dash(__name__, external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# app.layout = (
#   html.Div(children=dcc.Graph(figure=fig))
# )

# app.run_server()
