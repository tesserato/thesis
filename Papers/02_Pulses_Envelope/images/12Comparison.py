import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
import numpy.polynomial.polynomial as poly
import wave
from scipy.signal import savgol_filter, hilbert
import os
from pathlib import Path
import sys
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
from Wheel import frontiers
from plotly.subplots import make_subplots
from timeit import default_timer as timer



def read_wav(path): 
  """returns signal & fps"""
  wav = wave.open(path , 'r')
  signal = np.frombuffer(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps




'''==============='''
''' Read wav file '''
'''==============='''
name = "bend"
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
# '''    Sinusoid   '''
# '''==============='''
# FT = np.fft.rfft(W)# * 2 / n
# PW = np.abs(FT)
# f = np.argmax(PW)
# p = np.angle(FT[f])

# S = np.cos(p + 2 * np.pi * f * X / n)

'''==============='''
'''    Snowball   '''
'''==============='''
start = timer()
pos_frontierX, pos_frontierY, neg_frontierX, neg_frontierY = frontiers(W)

frontierX = np.concatenate([pos_frontierX, neg_frontierX])
frontierY = np.concatenate([pos_frontierY, np.abs(neg_frontierY)])

idxs = np.argsort(frontierX)
frontierX = frontierX[idxs]
frontierY = frontierY[idxs]

f = interp1d(frontierX, frontierY, kind="linear", fill_value="extrapolate", assume_sorted=False)

smooth_frontierY = savgol_filter(f(X), 5000 + 1, 2)

lms = np.average((np.abs(W) - smooth_frontierY * 0.5)**2)

envX = [x for x in X]
envY = [y for y in smooth_frontierY]
envX = envX + [None] + envX
envY = envY + [None] + [-y for y in envY]

print(f"This work: lms ={lms}, time={timer() - start}")
'''==============='''
'''   Smoothing   '''
'''==============='''
start = timer()
envY_smooth = savgol_filter(np.abs(W), 3000 + 1, 3)
envY_smooth = np.abs(envY_smooth) / np.max(np.abs(envY_smooth))
lms_smooth = np.average((np.abs(W) - envY_smooth * 0.5)**2)
envY_smooth = [y for y in envY_smooth] + [None] + [-y for y in envY_smooth]

print(f"Smoothing: lms ={lms_smooth}, time={timer() - start}")
'''==============='''
'''    Lowpass    '''
'''==============='''
start = timer()
FT = np.fft.rfft(W)

FT[200 :] = (0.0 + 0.0 * 1j)
# # FT = FT / np.max(FT)

W_lowpass = np.fft.irfft(FT)


sign = np.sign(W_lowpass[0])
x = 1
while np.sign(W_lowpass[x]) == sign:
  x += 1
x0 = x + 1
sign = np.sign(W_lowpass[x0])

posX = []
posY = []
negX = []
negY = []
for x in range(x0, W_lowpass.size):
  if np.sign(W_lowpass[x]) != sign: # Prospective pulse
    if x - x0 > 2:          # Not noise
      xp = x0 + np.argmax(np.abs(W[x0 : x]))
      yp = W_lowpass[xp]
      if np.sign(yp) >= 0:
        posX.append(xp)
        posY.append(yp)
      else:
        negX.append(xp)
        negY.append(yp)
    x0 = x
    sign = np.sign(W[x])

posX, posY, negX, negY = np.array(posX), np.array(posY), np.array(negX), np.array(negY)

frontierX = np.concatenate([posX, negX])
frontierY = np.concatenate([posY, np.abs(negY)])

idxs = np.argsort(frontierX)
frontierX = frontierX[idxs]
frontierY = frontierY[idxs]

f = interp1d(frontierX, frontierY, kind="linear", fill_value="extrapolate", assume_sorted=False)

# envY_lowpass = savgol_filter(f(X), 50000 + 1, 2)
envY_lowpass = savgol_filter(f(X), 5000 + 1, 2)
envY_lowpass = envY_lowpass / np.max(np.abs(envY_lowpass))
lms_lowpass = np.average((np.abs(W) - envY_lowpass * 0.5)**2)
envY_lowpass = [y for y in envY_lowpass] + [None] + [-y for y in envY_lowpass]

print(f"Lowpass:   lms ={lms_lowpass}, time={timer() - start}")
'''==============='''
'''    Hilbert    '''
'''==============='''
start = timer()
analytic_signal = hilbert(savgol_filter(np.abs(W), 3000 + 1, 3))
envY_hilbert = np.abs(analytic_signal)
lms_hilbert = np.average((np.abs(W) - envY_hilbert * 0.5)**2)
envY_hilbert = [y for y in envY_hilbert] + [None] + [-y for y in envY_hilbert]

print(f"Hilbert:   lms ={lms_hilbert}, time={timer() - start}")
# exit()
'''============================================================================'''
'''                                    PLOT FT                                    '''
'''============================================================================'''
fig = make_subplots( rows=1, cols=1 )

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

fig.update_xaxes(row=1, col=1, title_text="$i$", showline=False, showgrid=False, zeroline=False)#, range=[0, 5000])
fig.update_yaxes(row=1, col=1, title_text="Amplitude", showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver')#, range=[0, 0.005])


fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(width=.5, color="silver"),
  ),
  row=1, col=1
)


fig.add_trace(
  go.Scatter(
    name="Present Work", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=envX,
    y=envY,
    mode="lines",
    line=dict(width=1, color="black"),
  ),
  row=1, col=1
)


fig.add_trace(
  go.Scatter(
    name="Smoothing", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=envX,
    y=envY_smooth,
    mode="lines",
    line=dict(width=1, color="black", dash='dot'),
  ),
  row=1, col=1
)


fig.add_trace(
  go.Scatter(
    name="Lowpass Filter", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=envX,
    y=envY_lowpass,
    mode="lines",
    line=dict(width=1, color="dimgray", dash='solid'),
  ),
  row=1, col=1
)

fig.add_trace(
  go.Scatter(
    name="Hilbert", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=envX,
    y=envY_hilbert,
    mode="lines",
    line=dict(width=1, color="dimgray", dash='dot'),
  ),
  row=1, col=1
)


# fig.show(config=dict({'scrollZoom': True}))

save_name = "./" + sys.argv[0].split('/')[-1].replace(".py", ".pdf")
fig.write_image(save_name, width=800, height=400, scale=1, engine="kaleido")
print("saved:", save_name)