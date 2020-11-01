import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
import numpy.polynomial.polynomial as poly
import wave
from scipy.signal import savgol_filter
import os
from pathlib import Path
import sys
import argparse

def read_wav(path): 
  """returns signal & fps"""
  wav = wave.open(path , 'r')
  signal = np.frombuffer(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps


def get_circle(x0, y0, x1, y1, r):
  '''returns center of circle that passes through two points'''
  
  radsq = r * r
  q = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

  # assert q <= 2 * r, f"shit"

  x3 = (x0 + x1) / 2
  y3 = (y0 + y1) / 2

  if y0 + y1 >= 0:
    xc = x3 + np.sqrt(radsq - (q / 2)**2) * ((y0 - y1) / q)
    yc = y3 + np.sqrt(radsq - (q / 2)**2) * ((x1 - x0) / q)
  else:
    xc = x3 - np.sqrt(radsq - (q / 2)**2) * ((y0 - y1) / q)
    yc = y3 - np.sqrt(radsq - (q / 2)**2) * ((x1 - x0) / q)

  return xc, yc


def get_radius_function(X, Y):
  m0 = np.average(Y[1:] - Y[:-1]) / np.average(X[1:] - X[:-1])
  curvatures_X = []
  curvatures_Y = []
  for i in range(len(X) - 1):
    m1 = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])

    theta = np.arctan( (m1 - m0) / (1 + m1 * m0) )
    k = np.sin(theta) / (X[i + 1] - X[i])

    # if k >= 0:
    curvatures_X.append((X[i + 1] + X[i]) / 2)
    curvatures_Y.append(k)
    
  curvatures_Y = np.array(curvatures_Y)
  coefs = poly.polyfit(curvatures_X, curvatures_Y, 0)
  smooth_curvatures_Y = poly.polyval(curvatures_X, coefs)

  r_of_x = interp1d(curvatures_X, 1 / np.abs(smooth_curvatures_Y), fill_value="extrapolate")

  return r_of_x


def get_frontier(X, Y):
  '''extracts the envelope via snowball method'''
  scaling = (np.average(X[1:]-X[:-1]) / 2) / np.average(Y)
  Y = Y * scaling

  r_of_x = get_radius_function(X, Y)
  idx1 = 0
  idx2 = 1
  frontierX = [X[0]]
  frontierY = [Y[0]]
  n = len(X)
  while idx2 < n:
    r = r_of_x((X[idx1] + X[idx2]) / 2)
    xc, yc = get_circle(X[idx1], Y[idx1], X[idx2], Y[idx2], r) # Triangle
    empty = True
    for i in range(idx2 + 1, n):
      if np.sqrt((xc - X[i])**2 + (yc - Y[i])**2) < r:
        empty = False
        idx2 += 1
        break
    if empty:
      frontierX.append(X[idx2])
      frontierY.append(Y[idx2])
      idx1 = idx2
      idx2 += 1
  # frontierX.append(X[-1])
  # frontierY.append(Y[-1])
  frontierX = np.array(frontierX)
  frontierY = np.array(frontierY) / scaling
  return frontierX, frontierY

'''Creating Parser'''
parser = argparse.ArgumentParser()

parser.add_argument(
  "--name",
  type=str,
)

parser.add_argument(
  "--x0",
  type=int,
)

parser.add_argument(
  "--x1",
  type=int,
)

parser.add_argument(
  "--y0",
  type=float,
)

parser.add_argument(
  "--y1",
  type=float,
)

parser.add_argument(
  "--detailed",
  type=bool,
  default=False
)


args = vars(parser.parse_args())
print(f"Original args: {args}")
name = args["name"]
detailed = args["detailed"]
x0_plot = args["x0"]
x1_plot = args["x1"]
y0_plot = args["y0"]
y1_plot = args["y1"]
print(f"name:{name} detailed:{detailed} x0:{x0_plot} x1:{x1_plot} y0:{y0_plot} y1:{y1_plot}")


'''==============='''
''' Read wav file '''
'''==============='''
fig = go.Figure()

parent_path = str(Path(os.path.abspath('./')).parents[1])
path = f"{parent_path}/Python/Samples/{name}.wav"
print(path)

W, fps = read_wav(path)

W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
print(f"n={n}")
X = np.arange(n)


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

posX, posY, negX, negY = np.array(posX), np.array(posY), np.array(negX), np.array(negY)

pos_frontierX, pos_frontierY = get_frontier(posX, posY)
neg_frontierX, neg_frontierY = get_frontier(negX, negY)

# frontierX = np.concatenate([pos_frontierX, neg_frontierX])
# frontierY = np.concatenate([pos_frontierY, np.abs(neg_frontierY)])

# idxs = np.argsort(frontierX)
# frontierX = frontierX[idxs]
# frontierY = frontierY[idxs]

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''

fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="$i$",
  yaxis_title="Amplitude",

  # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
  # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Latin Modern Roman",
  color="black",
  size=18
  )
)

if detailed:
  fig.update_xaxes(showline=False, showgrid=False, zeroline=False, range=[x0_plot, x1_plot])
  fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver', range=[y0_plot, y1_plot])
else:
  fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
  fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver')


fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(
        # size=8,
        color="gray",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Frontiers", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=pos_frontierX,
    y=pos_frontierY,
    mode="lines",
    line=dict(
        width=3,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Negative Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=neg_frontierX,
    y=neg_frontierY,
    mode="lines",
    line=dict(
        width=3,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
    showlegend=False
  )
)

if not detailed:
  fig.add_trace(
    go.Scatter(
      name="Limits", # <|<|<|<|<|<|<|<|<|<|<|<|
      x=[x0_plot, x1_plot],
      y=[0, 0],
      mode="markers",
      marker=dict(
        size=1,
        color="black",
        # showscale=False
      ),
      # visible = "legendonly"
      showlegend=False
    )
  )

  fig.show(config=dict({'scrollZoom': True}))


if detailed:
  suffix = "_d"
  height = 100
  width = 400
else:
  suffix = ""
  height = 300
  width = 800


save_name = "./images/07FullFrontiers/" + sys.argv[0].split('\\')[-1].replace(".py", "") + "-" + name + suffix + ".svg"
fig.write_image(save_name, width=width, height=height, scale=1, engine="kaleido")
print("saved:", save_name)

# _ = input("PRESS ENTER TO CONTINUE.")