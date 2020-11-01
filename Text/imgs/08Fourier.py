import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
import numpy.polynomial.polynomial as poly
import wave
from scipy.signal import savgol_filter
import os
from pathlib import Path
import sys
# import argparse
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
name = "piano33"
parent_path = str(Path(os.path.abspath('./')).parents[0])
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

frontierX = np.concatenate([pos_frontierX, neg_frontierX])
frontierY = np.concatenate([pos_frontierY, np.abs(neg_frontierY)])

idxs = np.argsort(frontierX)
frontierX = frontierX[idxs]
frontierY = frontierY[idxs]

f = interp1d(frontierX, frontierY, fill_value="extrapolate", assume_sorted=True)

FT = np.abs(np.fft.rfft(W) * 2 / n)
# FT = FT / np.max(FT)

normalized_W = W / f(X)
normalized_W = normalized_W / np.average(np.abs(normalized_W)) * np.average(np.abs(W))
normalized_W = normalized_W - np.average(normalized_W)
normalized_FT = np.abs(np.fft.rfft(normalized_W) * 2 / n)
# normalized_FT = normalized_FT / np.max(normalized_FT)


'''============================================================================'''
'''                                    PLOT FT                                    '''
'''============================================================================'''
FONT = dict(
  family="Latin Modern Roman",
  color="black",
  size=13.3333
)

fig = make_subplots(
  rows=2, cols=1,
  subplot_titles=("Frequency Domain", "Time Domain")
)

fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="<b><i>i</i></b>",
  yaxis_title="<b>Amplitude</b>",
  paper_bgcolor='rgba(0,0,0,0)',
  plot_bgcolor='rgba(0,0,0,0)',
  legend=dict(orientation='h', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT
)
fig.update_xaxes(title_font = FONT, row=2, col=1, title_text="<b>Frequency</b>", showline=False, showgrid=False, zeroline=False, range=[0, 5000])
fig.update_yaxes(title_font = FONT, row=2, col=1, title_text="<b>Power</b>", showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver', range=[0, 0.005])


fig.add_trace(
  go.Scatter(
    name="Original Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=FT,
    mode="none",
    fill="tozeroy",
    fillcolor="black",
  ),
  row=2, col=1
)

fig.add_trace(
  go.Scatter(
    name="Normalized Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=normalized_FT,
    mode="none",
    fill="tozeroy",
    fillcolor="rgba(128,128,128,0.9)",
  ),
  row=2, col=1
)



'''============================================================================'''
'''                                    PLOT SIGNAL                                   '''
'''============================================================================'''
fig.update_xaxes(title_font = FONT, row=1, col=1, title_text="<b><i>i</i></b>", showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(title_font = FONT, row=1, col=1, title_text="<b>Amplitude</b>", showline=False, showgrid=False, zerolinewidth=1)


fig.add_trace(
  go.Scatter(
    name="Original Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    # line_shape='spline',
    line=dict(
      color="black",
      width=.1
      ),
    showlegend=False,
  ),
  row=1, col=1
)

fig.add_trace(
  go.Scatter(
    name="Normalized Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=normalized_W,
    mode="lines",
    # line_shape='spline',
    line=dict(
      color="rgba(128,128,128,0.9)",
      width=.1
    ),
    showlegend=False,
  ),
  row=1, col=1
)

fig.layout.annotations[0].update(x=0.875, font=FONT)
fig.layout.annotations[1].update(x=0.875, font=FONT)

# fig.show(config=dict({'scrollZoom': True}))

# fig.show()
save_name = "./imgs/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=605, height=400, engine="kaleido", format="svg")
print("saved:", save_name)
