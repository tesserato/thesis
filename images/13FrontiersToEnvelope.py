import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
# import numpy.polynomial.polynomial as poly
import wave
from scipy.signal import savgol_filter
import os
from pathlib import Path
import sys
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
# from Wheel import frontiers
import signal_envelope as se
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
pos_frontierX, neg_frontierX = se.get_frontiers(W)
pos_frontierY, neg_frontierY = W[pos_frontierX], W[neg_frontierX]

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
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=2, zerolinecolor='gray')



fig.add_trace(
  go.Scatter(
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=W,
    mode="lines",
    line=dict(width=.5, color="lightgray"),
  )
)


fig.add_trace(
  go.Scatter(
    name="Envelope", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=envX,
    y=envY,
    mode="lines",
    line=dict(width=1, color="black"),
  )
)

fig.add_trace(
  go.Scatter(
    name="Frontiers", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=pos_frontierX,
    y=pos_frontierY,
    mode="lines",
    line=dict(width=1, color="darkslategray", dash='solid'),
  )
)

fig.add_trace(
  go.Scatter(
    name="Positive Frontier Mirrored", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=pos_frontierX,
    y=-pos_frontierY,
    mode="lines",
    line=dict(width=1, color="gray", dash='solid'),
    showlegend=False
  )
)

fig.add_trace(
  go.Scatter(
    name="Negative Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=neg_frontierX,
    y=neg_frontierY,
    mode="lines",
    line=dict(width=1, color="darkslategray", dash='solid'),
    showlegend=False
  )
)

fig.add_trace(
  go.Scatter(
    name="Mirrored Frontiers", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=neg_frontierX,
    y=-neg_frontierY,
    mode="lines",
    line=dict(width=1, color="gray", dash='solid'),
    showlegend=True
  )
)

# fig.show()
save_name = "./imgs/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=605, height=300, engine="kaleido", format="svg")
print("saved:", save_name)