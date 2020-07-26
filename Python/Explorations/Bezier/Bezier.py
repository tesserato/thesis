import numpy as np
from numpy.core.function_base import linspace
import plotly.graph_objects as go
import numpy.polynomial.polynomial as poly
from scipy.interpolate import interp1d

def param_line_old_old(x0, x1, y0, y1):
  '''y = m x + b | 0 = x0, 1 = x1'''
  X = np.arange(x0, x1 + 1, 1)
  m = (y1 - y0) / (x1 - x0)
  b = y0 - m * x0
  f = lambda x : m * x + b
  Y = f(X)
  return X, Y

def param_line_old(x0, x1, y0, y1):
  '''y = m x + b | 0 = x0, 1 = x1'''
  T = np.linspace(0, 1, 20)
  X = x0 + T * (x1 - x0)
  m = (y1 - y0) / (x1 - x0)
  b = y0 - m * x0
  f = lambda x : m * x + b
  Y = f(X)
  return X, Y

def bezier_old_old(Xp, Yp):
  X = []
  Y = []
  for i in range(len(Xp) - 2):
    print(i)

    X_, Y_ = param_line(Xp[i], Xp[i + 1], Yp[i], Yp[i + 1])
    for x, y in zip(X_, Y_):
      X.append(x)
      Y.append(y)
    X.append(None)
    Y.append(None)

    X_, Y_ = param_line(Xp[i + 1], Xp[i + 2], Yp[i + 1], Yp[i + 2])
    for x, y in zip(X_, Y_):
      X.append(x)
      Y.append(y)
    X.append(None)
    Y.append(None)
  return X, Y

def bezier_old(Xp, Yp):
  X = []
  Y = []
  segments = len(Xp) - 2
  for i in range(segments):
    print(i)
    for t in np.linspace(0, 1, 20):
      x0, y0 = param_line(Xp[i], Xp[i + 1], Yp[i], Yp[i + 1], t)
      # X.append(x0)
      # Y.append(y0)

      x1, y1 = param_line(Xp[i + 1], Xp[i + 2], Yp[i + 1], Yp[i + 2], t)
      # X.append(x1)
      # Y.append(y1)

      # tt = 1 / ((2 * t - 1)**2 + 1)
      tt = (1.5 * (t - .5)) ** 2
      x, y = param_line(x0, x1, y0, y1, t)#i / segments + t / segments)
      X.append(x)
      Y.append(y)
      # X.append(None)
      # Y.append(None)

      # X.append((x0 + x1 + x) / 3)
      # Y.append((y0 + y1 + y) / 3)

  return X, Y

def param_line(x0, x1, y0, y1, t):
  '''y = m x + b | 0 = x0, 1 = x1'''
  T = np.linspace(0, 1, 20)
  x = x0 + t * (x1 - x0)
  m = (y1 - y0) / (x1 - x0)
  b = y0 - m * x0
  f = lambda alpha : m * alpha + b
  y = f(x)
  return x, y

def bezier(Xo, Yo):
  Xp = []
  Yp = []
  Xp.append(Xo[0])
  Yp.append(Yo[0])
  for i in range(1, len(Xo)-2):
    Xp.append(Xo[i]), Yp.append(Yo[i])
    Xp.append((Xo[i] + Xo[i + 1]) / 2), Yp.append((Yo[i] + Yo[i + 1]) / 2)
    # Xp.append(Xo[i + 1]), Yp.append(Yo[i + 1])
  Xp.append(Xo[-2])
  Yp.append(Yo[-2])
  Xp.append(Xo[-1])
  Yp.append(Yo[-1])
  print(Xo)
  print(Xp)

  X = []
  Y = []
  for i in range(0, len(Xp) - 2, 2):
    print(i)
    for t in np.linspace(0, 1, 20):
      x0, y0 = param_line(Xp[i], Xp[i + 1], Yp[i], Yp[i + 1], t)
      # X.append(x0)
      # Y.append(y0)

      x1, y1 = param_line(Xp[i + 1], Xp[i + 2], Yp[i + 1], Yp[i + 2], t)
      # X.append(x1)
      # Y.append(y1)

      # tt = 1 / ((2 * t - 1)**2 + 1)
      # tt = (1.5 * (t - .5)) ** 2
      x, y = param_line(x0, x1, y0, y1, t) # i / segments + t / segments)
      X.append(x)
      Y.append(y)
      # X.append(None)
      # Y.append(None)

      # X.append((x0 + x1 + x) / 3)
      # Y.append((y0 + y1 + y) / 3)

  return X, Y


def line(x0, x1, y0, y1, t):
  '''y = m x + b | 0 = x0, 1 = x1'''
  x = x0 + t * (x1 - x0)
  m = (y1 - y0) / (x1 - x0)
  b = y0 - m * x0
  f = lambda alpha : m * alpha + b
  y = f(x)
  return x, y

def bezier_approximation(Xo, Yo):
  Xp = []
  Yp = []
  Xp.append(Xo[0])
  Yp.append(Yo[0])
  for i in range(1, len(Xo)-2):
    Xp.append(Xo[i]), Yp.append(Yo[i])
    Xp.append((Xo[i] + Xo[i + 1]) / 2), Yp.append((Yo[i] + Yo[i + 1]) / 2)
  Xp.append(Xo[-2])
  Yp.append(Yo[-2])
  Xp.append(Xo[-1])
  Yp.append(Yo[-1])
  print(Xp)

  X = []
  Y = []
  for i in range(0, len(Xp) - 2, 2):

    x_0, x_1, x_2 = Xp[i], Xp[i + 1], Xp[i + 2]

    if x_0 + x_2 == 2 * x_1:
      print("oooooppsss!!!")
      # x_0 = x_0 + 1
      # x_1 = x_1 + 1
      x_2 = x_2 + 0.001

    int_x0 = int(np.ceil(x_0))
    int_x1 = int(np.floor(x_2))
    XX = np.arange(int_x0, int_x1 + 1, 1)

    T = (x_0 - x_1 + np.sqrt(XX * x_0 - 2 * XX * x_1 + XX * x_2 - x_0 * x_2 + x_1**2)) / (x_0 - 2 * x_1 + x_2)

    print("o:", Xp[i], Xp[i + 2])
    print("i:", int_x0, int_x1)
    print("x:", XX)
    print("t:", T, "\n")

    for t in T:
      x0, y0 = line(x_0, x_1, Yp[i], Yp[i + 1], t)
      # X.append(x0)
      # Y.append(y0)

      x1, y1 = line(x_1, x_2, Yp[i + 1], Yp[i + 2], t)
      # X.append(x1)
      # Y.append(y1)

      x, y = line(x0, x1, y0, y1, t) # i / segments + t / segments)
      X.append(x)
      Y.append(y)
    # X.append(None)
    # Y.append(None)

  return X, Y

def poly_approximation(Xo, Yo):
  Xp = []
  Yp = []
  Xp.append(Xo[0])
  Yp.append(Yo[0])
  for i in range(1, len(Xo)-2):
    Xp.append(Xo[i]), Yp.append(Yo[i])
    Xp.append((Xo[i] + Xo[i + 1]) / 2), Yp.append((Yo[i] + Yo[i + 1]) / 2)
  Xp.append(Xo[-2])
  Yp.append(Yo[-2])
  Xp.append(Xo[-1])
  Yp.append(Yo[-1])
  print(Xp)

  X = []
  Y = []
  for i in range(0, len(Xp) - 2, 2):

    x_0, x_1, x_2 = Xp[i], Xp[i + 1], Xp[i + 2]
    int_x0 = int(np.ceil(x_0))
    int_x1 = int(np.floor(x_2))
    XX = np.arange(int_x0, int_x1 + 1, 1)

    A = poly.polyfit(Xp[i:i+3], Yp[i:i+3], 2)
    YY = poly.polyval(XX, A)
    for x, y in zip(XX, YY):
      X.append(x)
      Y.append(y)

  return X, Y

def parabola_approximation(Xo, Yo):
  Xp = []
  Yp = []
  Xp.append(Xo[0])
  Yp.append(Yo[0])
  for i in range(1, len(Xo)-2):
    Xp.append(Xo[i]), Yp.append(Yo[i])
    Xp.append((Xo[i] + Xo[i + 1]) / 2), Yp.append((Yo[i] + Yo[i + 1]) / 2)
  Xp.append(Xo[-2])
  Yp.append(Yo[-2])
  Xp.append(Xo[-1])
  Yp.append(Yo[-1])
  print(Xp)

  X = []
  Y = []
  for i in range(0, len(Xp) - 2, 2):

    x_0, x_1, x_2 = Xp[i], Xp[i + 1], Xp[i + 2]


    int_x0 = int(np.ceil(x_0))
    int_x1 = int(np.floor(x_2))
    XX = np.arange(int_x0, int_x1 + 1, 1)

    print("o:", Xp[i], Xp[i + 2])
    print("i:", int_x0, int_x1)
    print("x:", XX)

    A = np.zeros((5 , 5))
    for j, t in enumerate(np.linspace(0, 1, 5)):
      x0, y0 = line(x_0, x_1, Yp[i], Yp[i + 1], t)
      x1, y1 = line(x_1, x_2, Yp[i + 1], Yp[i + 2], t)
      x, y = line(x0, x1, y0, y1, t)
      X.append(x)
      Y.append(y)
      A[j, 0] = x**2
      A[j, 1] = x * y
      A[j, 2] = y**2
      A[j, 3] = x
      A[j, 4] = y
    B = -np.ones((5)).T
    # print(A)
    coefs = np.linalg.solve(A, B)
    a, b, c, d, e = coefs
    print(coefs)
    # assert b**2==4*a*c
    # YY = (-b*XX - e + np.sqrt(-4*a*c*XX**2 + b**2*XX**2 + 2*b*e*XX - 4*c*d*XX - 4*c + e**2))/(2*c)
    YY = -(b*XX + e + np.sqrt(-4*a*c*XX**2 + b**2*XX**2 + 2*b*e*XX - 4*c*d*XX - 4*c + e**2))/(2*c)
    # for x, y in zip(XX, YY):
    #   X.append(x)
    #   Y.append(y)

  return X, Y

def bezier_approximation_interp(Xo, Yo):
  Xp = []
  Yp = []
  Xp.append(Xo[0])
  Yp.append(Yo[0])
  for i in range(1, len(Xo)-2):
    Xp.append(Xo[i]), Yp.append(Yo[i])
    Xp.append((Xo[i] + Xo[i + 1]) / 2), Yp.append((Yo[i] + Yo[i + 1]) / 2)
  Xp.append(Xo[-2])
  Yp.append(Yo[-2])
  Xp.append(Xo[-1])
  Yp.append(Yo[-1])
  print(Xp)

  X = []
  Y = []
  for i in range(0, len(Xp) - 3, 2):
    distance = np.sqrt((Xp[i + 2] - Xp[i])**2 + (Yp[i + 2] - Yp[i])**2)
    distance = np.int(np.ceil(distance))
    mt = 1 * (distance - 1) / distance
    for t in linspace(0, mt, distance):
      x0, y0 = line(Xp[i], Xp[i + 1], Yp[i], Yp[i + 1], t)
      x1, y1 = line(Xp[i + 1], Xp[i + 2], Yp[i + 1], Yp[i + 2], t)
      x, y = line(x0, x1, y0, y1, t)
      X.append(x)
      Y.append(y)
  distance = np.sqrt((Xp[-1] - Xp[-3])**2 + (Yp[-1] - Yp[-3])**2)
  distance = np.int(np.ceil(distance))
  for t in linspace(0, 1, distance):
    x0, y0 = line(Xp[-3], Xp[-2], Yp[-3], Yp[-2], t)
    x1, y1 = line(Xp[-2], Xp[-1], Yp[-2], Yp[-1], t)
    x, y = line(x0, x1, y0, y1, t)
    X.append(x)
    Y.append(y)

  f = interp1d(X, Y, kind="cubic", fill_value="extrapolate", assume_sorted=True)
  return X, Y, f



Xp = [0, 5, 9, 14, 20, 26, 30, 37]
Yp = [3, 2, 4, 1, 5, 2, 3, 1]

Xb, Yb, f = bezier_approximation_interp(Xp, Yp)

X = np.arange(np.max(Xp) + 1)
Y = f(X)


'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''

fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",
  # yaxis = dict(
  #   scaleanchor = "x",
  #   scaleratio = 1,
  # ),
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

fig.add_trace(
  go.Scatter(
    name="Control Points",
    x=Xp,
    y=Yp,
    mode="markers",
    marker=dict(
        size=8,
        color="red",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="Bezier",
    x=Xb,
    y=Yb,
    mode="markers+lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="Interpolation",
    x=X,
    y=Y,
    mode="markers+lines",
    line=dict(
        # size=8,
        color="blue",
        # showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': True}))