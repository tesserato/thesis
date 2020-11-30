import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
import numpy.polynomial.polynomial as poly
import wave
from scipy.signal import savgol_filter, hilbert
import os
from pathlib import Path
import sys
# sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
# from Wheel import frontiers
from plotly.subplots import make_subplots
from timeit import default_timer as timer
import signal_envelope as se




'''==============='''
''' Read wav file '''
'''==============='''
name = "bend"
W, fps = se.read_wav(f"C:/Users/tesse/Desktop/Files/Dropbox/0_Science/reps/envelope/test_samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
print(f"n={n}")
X = np.arange(n)

'''==============='''
'''    Snowball   '''
'''==============='''
start = timer()
Ex = se.get_frontiers(W, 1)


f = interp1d(Ex, np.abs(W[Ex]), kind="linear", fill_value="extrapolate", assume_sorted=False)
envY = f(X)

lms = np.average((np.abs(W) - envY * 0.5)**2)


print(f"This work: lms ={lms}, time={timer() - start}")
'''==============='''
'''   Smoothing   '''
'''==============='''
start = timer()
envY_smooth = savgol_filter(np.abs(W), 3000 + 1, 3)
envY_smooth = np.abs(envY_smooth) / np.max(np.abs(envY_smooth))
lms_smooth = np.average((np.abs(W) - envY_smooth * 0.5)**2)
# envY_smooth = [y for y in envY_smooth] + [None] + [-y for y in envY_smooth]

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
# envY_lowpass = [y for y in envY_lowpass] + [None] + [-y for y in envY_lowpass]

print(f"Lowpass:   lms ={lms_lowpass}, time={timer() - start}")
'''==============='''
'''    Hilbert    '''
'''==============='''
start = timer()
analytic_signal = hilbert(savgol_filter(np.abs(W), 3000 + 1, 3))
envY_hilbert = np.abs(analytic_signal)
lms_hilbert = np.average((np.abs(W) - envY_hilbert * 0.5)**2)
# envY_hilbert = [y for y in envY_hilbert] + [None] + [-y for y in envY_hilbert]

print(f"Hilbert:   lms ={lms_hilbert}, time={timer() - start}")
# exit()
'''============================================================================'''
'''                                    PLOT FT                                    '''
'''============================================================================'''
FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

'''Plotting'''
fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="<b><i>i</i></b>",
  yaxis_title="<b>Amplitude</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zeroline=False)



fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(width=.5, color="silver"),
  )
)


fig.add_trace(
  go.Scatter(
    name="Present Work", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=envY,
    mode="lines",
    line=dict(width=1, color="black"),
  )
)

fig.add_trace(
  go.Scatter(
    name="Smoothing", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=envY_smooth,
    mode="lines",
    line=dict(width=1, color="black", dash='dot'),
  )
)

fig.add_trace(
  go.Scatter(
    name="Lowpass Filter", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=envY_lowpass,
    mode="lines",
    line=dict(width=1, color="dimgray", dash='solid'),
  )
)

fig.add_trace(
  go.Scatter(
    name="Hilbert", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=envY_hilbert,
    mode="lines",
    line=dict(width=1, color="dimgray", dash='dot'),
  )
)


fig.show(config=dict({'scrollZoom': True}))

# save_name = "./" + sys.argv[0].split('/')[-1].replace(".py", ".pdf")
# fig.write_image(save_name, width=800, height=400, scale=1, engine="kaleido")
# print("saved:", save_name)