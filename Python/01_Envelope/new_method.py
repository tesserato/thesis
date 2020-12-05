import numpy as np
import wave
import plotly.graph_objects as go

fig = go.Figure()

def draw_circle(xc, yc, r, n=1000, color="red"):
  '''draws circle as plotly scatter from center and radius'''
  X = []
  Y = []
  for t in np.linspace(0, 2 * np.pi, n):
    X.append(xc + r * np.cos(t))
    Y.append(yc + r * np.sin(t))
  fig.add_trace(
    go.Scatter(
      name="Circles",
      # legendgroup="Circles",
      x=X,
      y=Y,
      # showlegend=False,
      visible = "legendonly",
      mode="lines",
      line=dict(
          width=3,
          color=color
      )
    )
  )
  return X, Y

def read_wav(path): 
  """returns signal & fps"""
  wav = wave.open(path , 'r')
  signal = np.frombuffer(wav.readframes(-1) , np.int16).astype(np.double)
  fps = wav.getframerate()
  return signal, fps

def save_wav(signal, name = 'test.wav', fps = 44100): 
  '''save .wav file to program folder'''
  o = wave.open(name, 'wb')
  o.setframerate(fps)
  o.setnchannels(1)
  o.setsampwidth(2)
  o.writeframes(np.int16(signal)) # Int16
  o.close()

def _get_circle(x0, y0, x1, y1, r):
  '''returns center of circle that passes through two points'''  
  q = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
  c = np.sqrt(r * r - (q / 2)**2)
  x3 = (x0 + x1) / 2
  y3 = (y0 + y1) / 2 
  if y0 + y1 >= 0:
    xc = x3 + c * (y0 - y1) / q
    yc = y3 + c * (x1 - x0) / q
  else:
    xc = x3 - c * (y0 - y1) / q
    yc = y3 - c * (x1 - x0) / q

  return xc, yc

def _get_pulses(W):
  '''Sorts a signal into pulses, returning positive and negative X, Y coordinates, and filtering out noise'''
  n = W.size
  sign = np.sign(W[0])
  x = 1
  while np.sign(W[x]) == sign:
    x += 1
  x0 = x + 1
  sign = np.sign(W[x0])
  posX = []
  posY = []
  negX = []
  negY = []
  for x in range(x0, n):
    if np.sign(W[x]) != sign: # Prospective pulse
      if x - x0 > 2:          # Not noise
        xp = x0 + np.argmax(np.abs(W[x0 : x]))
        yp = W[xp]
        if np.sign(yp) >= 0:
          posX.append(xp)
          posY.append(yp)
        else:
          negX.append(xp)
          negY.append(yp)
      x0 = x
      sign = np.sign(W[x])
  return np.array(posX), np.array(posY), np.array(negX), np.array(negY)

def _get_radius_average5(X, Y):
  m0 = 0 #(Y[-1] - Y[0]) / (X[-1] - X[0])
  k_sum = 0
  mm = np.sqrt(m0**2 + 1)                            
  for i in range(len(X) - 1):
    x = (X[i + 1] - X[i])                            
    y = (Y[i + 1] - Y[i])                            
    k = -(m0 * x - y) / (x * mm * np.sqrt(x*x + y*y))
    k_sum += k
  r = np.abs(1 / (k_sum / (len(X) - 1)))
  # print("m0: ", m0, "k: ", k_sum)
  print(f"radius={r}")
  return r

def _get_radius_average(X, Y):
  m0 = (Y[-1] - Y[0]) / (X[-1] - X[0])
  k_sum = 0
  mm = np.sqrt(m0**2 + 1)                            
  for i in range(len(X) - 1):
    x = (X[i + 1] - X[i])                            
    y = (Y[i + 1] - Y[i])                            
    k = -(m0 * x - y) / (x * mm * np.sqrt(x*x + y*y))
    k_sum += k
  r = np.abs(1 / (k_sum / (len(X) - 1)))
  # print("m0: ", m0, "k: ", k_sum)
  print(f"radius={r}")
  return r

def _get_radius_average3(X, Y):
  posVy = 0
  posVx = 0
  negVy = 0
  negVx = 0
  for i in range(X.size - 1): 
    vx = X[i + 1] - X[i]
    vy = Y[i + 1] - Y[i]
    if vy > 0 :
      posVy += vy #/ (X.size - 1)
      posVx += vx #/ (X.size - 1)
    else:
      negVy += vy #/ (X.size - 1)
      negVx += vx #/ (X.size - 1)

  vector_2 = [negVx, negVy]
  vector_1 = [posVx, posVy]

  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  angle = np.arccos(dot_product)  
  m0 = np.tan(angle)

  r = np.abs(1 / (k_sum / (len(X) - 1)))
  print("m0: ", m0, "k: ", k_sum)
  print(f"radius={r}")
  return r

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    print(radius)
    return radius

def _get_radius_average2(X, Y):
  posVy = 0
  posVx = 0
  negVy = 0
  negVx = 0
  for i in range(X.size - 1): 
    vx = X[i + 1] - X[i]
    vy = Y[i + 1] - Y[i]
    if vy > 0 :
      posVy += vy / (X.size - 1)
      posVx += vx / (X.size - 1)
    else:
      negVy += vy / (X.size - 1)
      negVx += vx / (X.size - 1)

  return 2 * define_circle((0, 0), (negVx, negVy), (negVx + posVx, negVy + posVy) )


def _get_radius_average4(X, Y):
  posVy = 0
  posVx = 0
  negVy = 0
  negVx = 0
  for i in range(X.size - 1): 
    vx = X[i + 1] - X[i]
    vy = Y[i + 1] - Y[i]
    if vy > 0 :
      posVy += vy
      posVx += vx
    else:
      negVy += vy
      negVx += vx

  # vector_1 = [posVx / (negVx + posVx) * (n), posVy / (negVx + posVx) * (n)]
  # vector_2 = [negVx / (negVx + posVx) * (n), negVy / (negVx + posVx) * (n)]

  vector_0 = [negVx / (X.size - 1), negVy / (X.size - 1)]
  vector_1 = [posVx / (X.size - 1), posVy / (X.size - 1)]

  ap0 = -1 / (vector_0[1] / vector_0[0])
  ap1 = -1 / (vector_1[1] / vector_1[0])

  b1 = vector_1[1] - (vector_1[0] + vector_0[0]) * ap1
  xc = b1 / (ap0 - ap1)
  yc = ap0 * xc
  r = np.sqrt((xc)**2 + (yc)**2)

  "y = ap0 x"
  "y = ap1 x + b1"
  "ap0 x - ap1 x = b1"

  fig.add_trace(
    go.Scatter(
      name="vecs",
      x=[0, vector_0[0], vector_1[0] + vector_0[0]],
      y=[0, vector_0[1], vector_1[1] + vector_0[1]],
      mode="lines",
      line=dict(color="red",),
    )
  )

  fig.add_trace(
    go.Scatter(
      name="C",
      x=[0, xc, vector_1[0] + vector_0[0]],
      y=[0, yc, vector_1[1] + vector_0[1]],
      mode="lines",
      line=dict(color="black",),
    )
  )

  draw_circle(xc, yc, r)

  return r #define_circle((0, 0), vector_2, (vector_1[0] + vector_2[0], vector_1[1] + vector_2[1]) )


def _get_frontier(X, Y):
  '''extracts the frontier via snowball method'''
  avgL = np.average((X[1:] - X[:-1]) / 2)
  avgY = np.average(Y)

  # avgL = np.average((X[1:] - X[:-1]) / 2)
  # avgY = np.average(np.abs(Y[1:] - Y[:-1]))
  Y = avgL * Y / avgY
  
  r = _get_radius_average5(X, Y)
  idx1 = 0
  idx2 = 1
  frontierX = [X[0]]
  # frontierY = [Y[0]]
  n = len(X)
  # print("n: ",n, " r: ", r_of_x(0)," Y0: ", Y[0])
  while idx2 < n:
    xc, yc = _get_circle(X[idx1], Y[idx1], X[idx2], Y[idx2], r)
    empty = True
    for i in range(idx2 + 1, n):
      if np.sqrt((xc - X[i])**2 + (yc - Y[i])**2) < r:
        empty = False
        idx2 += 1
        break
    if empty:
      frontierX.append(X[idx2])
      # frontierY.append(Y[idx2])
      idx1 = idx2
      idx2 += 1
  frontierX = np.array(frontierX)
  # frontierY = np.array(frontierY) / scaling
  # print(frontierX.size)
  return frontierX




def get_frontiers_py(W):
  "Returns positive and negative frontiers of a signal"
  PosX, PosY, NegX, NegY = _get_pulses(W)
  X = np.hstack([PosX, NegX])
  # Y = np.hstack([PosY, np.abs(NegY)])
  X = np.unique(X)
  PosFrontierX = _get_frontier(PosX, W[PosX])
  return PosFrontierX

W, _ = read_wav("Samples/brass.wav")
 
# posX, posY, negX, negY = _get_pulses(W)

# L = posX[1:] - posX[:-1]
# avgL = np.average(L)# + np.std(L)
# avgY = np.average(np.abs(posY))# + np.std(np.abs(posY))
# W = avgL * W / avgY
n = W.size

Xpos = get_frontiers_py(W)

# PosX, PosY, NegX, NegY = _get_pulses(W)




'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''

# import plotly.graph_objects as go

fig.layout.template ="plotly_white"
fig.update_layout(
  xaxis_title="x",
  yaxis_title="Amplitude",

  # yaxis = dict(scaleanchor = "x",scaleratio = 1),

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
    name="Signal",
    # x=np.arange(W.size),
    y=W,
    mode="lines",
    line=dict(color="silver",),
  )
)

fig.add_trace(
  go.Scatter(
    name="+Frontier",
    x=Xpos,
    y=np.abs(W[Xpos]),
    mode="lines",
    line=dict(width=1, color="red"),
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="-Frontier",
#     x=Xneg,
#     y=W[Xneg],
#     mode="lines",
#     line=dict(width=1, color="red"),
#   )
# )

fig.show(config=dict({'scrollZoom': True}))
