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
from plotly.subplots import make_subplots

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


'''==============='''
''' Read wav file '''
'''==============='''
name = "tom"
parent_path = str(Path(os.path.abspath('./')).parents[1])
path = f"{parent_path}/Python/Samples/{name}.wav"
print(path)

W, fps = read_wav(path)

W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
print(f"n={n}, amplitude={amplitude}")
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

#10000000000000000000
#1.0e-305
scale = 1
pos_frontierX_scaled, pos_frontierY_scaled = pos_frontierX, pos_frontierY * scale
neg_frontierX_scaled, neg_frontierY_scaled = neg_frontierX, neg_frontierY * scale
while np.allclose(pos_frontierY_scaled / scale, pos_frontierY) and np.allclose(neg_frontierY_scaled / scale, neg_frontierY):
  print(scale)
  scale = scale * 10
  pos_frontierX_scaled, pos_frontierY_scaled = get_frontier(posX, posY * scale)
  neg_frontierX_scaled, neg_frontierY_scaled = get_frontier(negX, negY * scale)

'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01
    # subplot_titles=("Frequency Domain", "Time Domain")
    )

fig.layout.template ="plotly_white"
fig.update_layout(
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Latin Modern Roman",
  color="black",
  size=18
  )
)

fig.update_xaxes(row=1, col=1, showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(row=1, col=1, title_text="Normalized Amplitude", showline=False, showgrid=False, zerolinewidth=1, zerolinecolor="silver")

fig.add_trace(
  go.Scatter(
    name="Original Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(color="silver"),
    showlegend=False,
  ),
  row=1, col=1
)

fig.add_trace(
  go.Scatter(
    name="Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=pos_frontierX,
    y=pos_frontierY,
    mode="lines",
    line=dict(color="Black"),
    showlegend=False,
  ),
  row=1, col=1
)

fig.add_trace(
  go.Scatter(
    name="Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=neg_frontierX,
    y=neg_frontierY,
    mode="lines",
    line=dict(color="Black"),
    showlegend=False,
  ),
  row=1, col=1
)

pos_frontierX_scaled, pos_frontierY_scaled = get_frontier(posX, posY * scale)
neg_frontierX_scaled, neg_frontierY_scaled = get_frontier(negX, negY * scale)

fig.update_xaxes(row=2, col=1, title_text="$i$", showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(row=2, col=1, title_text="Max Amplitude", showline=False, showgrid=False, zerolinewidth=1, zerolinecolor="silver")

fig.add_trace(
  go.Scatter(
    name="Original Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W * scale,
    mode="lines",
    line=dict(color="silver"),
    showlegend=False,
  ),
  row=2, col=1
)

fig.add_trace(
  go.Scatter(
    name="Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=pos_frontierX_scaled,
    y=pos_frontierY_scaled,
    mode="lines",
    line=dict(color="Black"),
    showlegend=False,
  ),
  row=2, col=1
)

fig.add_trace(
  go.Scatter(
    name="Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=neg_frontierX_scaled,
    y=neg_frontierY_scaled,
    mode="lines",
    line=dict(color="Black"),
    showlegend=False,
  ),
  row=2, col=1
)

# fig.layout.annotations[0].update(x=0.875)

fig.show(config=dict({'scrollZoom': True}))

# save_name = "./" + sys.argv[0].split('/')[-1].replace(".py", ".pdf")
# fig.write_image(save_name, width=800, height=400, scale=1, engine="kaleido")
# print("saved:", save_name)