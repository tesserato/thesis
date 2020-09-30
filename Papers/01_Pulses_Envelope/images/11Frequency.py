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
name = "alto"
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

# '''==============='''
# ''' Make wav file '''
# '''==============='''
# n = 44100
# X = np.arange(n)
# fo = 5
# po = 3
# W = np.cos(po + 2 * np.pi * fo * X / n)


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

FT = np.fft.rfft(W) * 2 / n
PW = np.abs(FT)
f = np.argmax(PW)
p = np.angle(FT[f])

f_pf = (pos_frontierX.size - 1) * n / (pos_frontierX[-1] - pos_frontierX[0])
p_pf = np.average((2 * np.pi - 2 * f_pf * pos_frontierX * np.pi / n) % (2 * np.pi))

f_nf = (neg_frontierX.size - 1) * n / (neg_frontierX[-1] - neg_frontierX[0])
p_nf = np.average((np.pi - 2 * f_nf * neg_frontierX * np.pi / n) % (2 * np.pi))

print(f"FT: f={f}, p={round(p, 2)} PF: f={f_pf}, p={p_pf} NF: f={f_nf}, p={p_nf}")

exit()




'''============================================================================'''
'''                                    PLOT FT                                    '''
'''============================================================================'''
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Frequency Domain", "Time Domain")
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

fig.update_xaxes(row=2, col=1, title_text="Frequency", showline=False, showgrid=False, zeroline=False, range=[0, 5000])
fig.update_yaxes(row=2, col=1, title_text="Power", showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver', range=[0, 0.005])


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

fig.update_xaxes(row=1, col=1, title_text="$i$", showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(row=1, col=1, title_text="Amplitude", showline=False, showgrid=False, zerolinewidth=1)


fig.add_trace(
  go.Scatter(
    name="Original Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(color="black"),
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
    line=dict(color="rgba(128,128,128,0.9)"),
    showlegend=False,
  ),
  row=1, col=1
)

fig.layout.annotations[0].update(x=0.875)
fig.layout.annotations[1].update(x=0.875)

# fig.show(config=dict({'scrollZoom': True}))

save_name = "./" + sys.argv[0].split('/')[-1].replace(".py", ".pdf")
fig.write_image(save_name, width=800, height=400, scale=1, engine="kaleido")
print("saved:", save_name)