import plotly.graph_objects as go
import numpy as np

from plotly.colors import DEFAULT_PLOTLY_COLORS
r = np.array([[0, -1], [1, 0]])  # rotation matrix
fig = go.Figure()

fig.update_layout(
  # width = np.pi * 440,
  # height = n * 440,
  yaxis = dict(scaleanchor = "x", scaleratio = 1),
  # title=f"n = {n}",
  # xaxis_title="Phase",
  # yaxis_title="Frequency",
  # font=dict(
  #     family="Courier New, monospace",
  #     size=18,
  #     color="#7f7f7f"  )
  )

class Line:
  Counter = 1
  Lines = []

  @classmethod
  def Plot(cls):
    for line in cls.Lines:
      line.plot()

  @classmethod
  def Intersection(cls):
    Sn  = np.zeros((2, 2))
    Snp = np.zeros((2, 1))
    for line in cls.Lines:
      nnT = line.n @ line.n.T
      Sn  += nnT #* line.w
      Snp += nnT @ line.p #* line.w
    return np.linalg.inv(Sn) @ Snp

  def __init__(self, x_in_L, y_in_L, x_V, y_V, weight=1):
    self.number = Line.Counter
    Line.Counter += 1
    Line.Lines.append(self)
    self.xl = x_in_L
    self.yl = y_in_L
    self.xv = x_V
    self.yv = y_V
    self.w  = weight

    self.p = np.array([[self.xl], [self.yl]])    # point in l1
    self.v = np.array([[self.xv], [self.yv]])    # vector perpendicular to l1
    self.n = self.v / np.sqrt(self.v.T @ self.v) # unit vector perpendicular to l1
    self.u = r @ self.n                          # unit vector in the direction of l1

  def plot(self, xi=-10, xf=10):
    color = DEFAULT_PLOTLY_COLORS[(self.number - 1) % len(DEFAULT_PLOTLY_COLORS)]
    # plotting point
    fig.add_trace(go.Scatter(name=f"point {self.number}",
      legendgroup=f"l={self.number}",
      showlegend=False,
      x=[self.xl],
      y=[self.yl],
      mode='markers',
      marker_symbol="circle-open",
      marker_line_width=2,
      marker=dict(size=15, color=color, showscale=False)))

    # plotting line from x=xi to x=xf
    li = (xi - self.xl) / self.u[0, 0]
    lf = (xf - self.xl) / self.u[0, 0]
    yi = (self.p + li * self.u)[1, 0]
    yf = (self.p + lf * self.u)[1, 0]
    fig.add_trace(go.Scatter(name=f"line {self.number}",
      legendgroup=f"l={self.number}",
      x=[xi, xf], 
      y=[yi, yf],
      line = dict(color=color),
      mode="lines"))

    # plotting V
    # line
    fig.add_trace(go.Scatter(
      legendgroup=f"l={self.number}",
      showlegend=False,
      x=[0, self.xv], 
      y=[0, self.yv],
      line = dict(color=color, dash='dash'),
      mode="lines"))
    # plotting head
    fig.add_trace(go.Scatter(name=f"V {self.number}",
      legendgroup=f"l={self.number}",
      showlegend=False,
      x=[self.xv],
      y=[self.yv],
      mode='markers',
      marker=dict(color=color)))

    # plotting N
    # line
    fig.add_trace(go.Scatter(
      legendgroup=f"l={self.number}",
      showlegend=False,
      x=[0, self.n[0, 0]], 
      y=[0, self.n[1, 0]],
      line = dict(color=color),
      mode="lines"))
    # plotting head
    fig.add_trace(go.Scatter(name=f"N {self.number}",
      legendgroup=f"l={self.number}",
      showlegend=False,
      x=[self.n[0, 0]],
      y=[self.n[1, 0]],
      mode='markers',
      marker=dict(color=color)))

    # plotting U
    # line
    fig.add_trace(go.Scatter(
      legendgroup=f"l={self.number}",
      showlegend=False,
      x=[0, self.u[0, 0]], 
      y=[0, self.u[1, 0]],
      line = dict(color=color),
      mode="lines"))
    # plotting head
    fig.add_trace(go.Scatter(name=f"U {self.number}",
      legendgroup=f"l={self.number}",
      showlegend=False,
      x=[self.u[0, 0]],
      y=[self.u[1, 0]],
      mode='markers',
      marker=dict(color=color)))

Line(4,3,1,3,   1)
Line(5,4,2,3,   1)
Line(9,15,-3,4, 1)

# Line(7,15,0,4)
# Line(7,12,0,-4)

Line.Plot()

p = Line.Intersection()

fig.add_trace(go.Scatter(name=f"intersection",
  x=[p[0, 0]],
  y=[p[1, 0]],
  mode='markers',
  marker_symbol="circle-open-dot",
  marker_line_width=2,
  marker=dict(size=15, color="black")))

fig.show(config=dict({'scrollZoom': False}))


