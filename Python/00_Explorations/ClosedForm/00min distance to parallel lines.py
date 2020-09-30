# import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

    fig.add_trace(go.Scatter(name=f"t={self.t}",
      legendgroup=f"t={self.t}",
      x=X,
      y=Y,
      line = dict(color=color),
      mode="lines"))

    # plotting V
    # line
    fig.add_trace(go.Scatter(name=f"V {self.number}",
      legendgroup=f"t={self.t}",
      showlegend=False,
      x=[0, self.xv], 
      y=[0, self.yv],
      line = dict(color=color, dash='dash'),
      mode="lines"))
    # plotting head
    fig.add_trace(go.Scatter(name=f"point {self.number}",
      legendgroup=f"t={self.t}",
      showlegend=False,
      x=[self.xv],
      y=[self.yv],
      mode='markers',
      marker=dict(color=color)))

    # plotting N
    # line
    fig.add_trace(go.Scatter(name=f"V {self.number}",
      legendgroup=f"t={self.t}",
      showlegend=False,
      x=[0, self.n[0, 0]], 
      y=[0, self.n[1, 0]],
      line = dict(color=color), mode="lines"))
    # plotting head
    fig.add_trace(go.Scatter(name=f"PV {self.number}",
      legendgroup=f"t={self.t}",
      showlegend=False,
      x=[self.n[0, 0]],
      y=[self.n[1, 0]],
      mode='markers',
      marker=dict(color=color)))

    # plotting U
    # line
    fig.add_trace(go.Scatter(name=f"U {self.number}",
      legendgroup=f"t={self.t}",
      showlegend=False,
      x=[0, self.u[0, 0]], 
      y=[0, self.u[1, 0]],
      line = dict(color=color), mode="lines"))
    # plotting head
    fig.add_trace(go.Scatter(name=f"PU {self.number}",
      legendgroup=f"t={self.t}",
      showlegend=False,
      x=[self.u[0, 0]],
      y=[self.u[1, 0]],
      mode='markers',
      marker=dict(color=color)))


#######################
#######################
n = 20
X = np.arange(n)
f = 5.5
p = 1.5 * np.pi
W = np.cos(p + 2 * np.pi * f * X / n)
#######################
#######################

fig = go.Figure()
fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
fig.update_yaxes(tickvals=[i for i in range(-n, n+1)])

fig.update_layout(
  height = 2000,
  yaxis = dict(scaleanchor = "x", scaleratio = 1),
  xaxis_title="Phase",
  yaxis_title="Frequency")

ParallelLines(15, n)
ParallelLines(16, n)
ParallelLines(17, n)
ParallelLines(18, n)

ParallelLines.Plot(ymax=n+1)

fig.show(config=dict({'scrollZoom': False}))



