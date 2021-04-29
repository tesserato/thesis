import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, hilbert, butter, filtfilt
import sys
from timeit import default_timer as timer
import signal_envelope as se

def butter_lowpass_filter(data, fps, cutoff = 10, order = 2):
  nyq = 0.5 * fps
  normal_cutoff = cutoff / nyq
  # Get the filter coefficients 
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = filtfilt(b, a, data)
  return y

def get_reference_envelope(W):
  Wp = np.pad(W, (0, n//2))
  Wp_roll = np.copy(Wp)
  conv = np.zeros(n//2)

  t = int(n / np.argmax(np.abs(np.fft.rfft(W))))

  for i in range(t//2, 10 * t):
    Wp_roll = np.roll(Wp_roll, 1)
    conv[i] = np.sum(Wp_roll * Wp)

  t = int(np.argmax(conv) * 2)
  rolled_W = np.roll(np.pad(W, (t//2, t//2) ), -t//2)

  # print(t)
  for i in range(t):
    rolled_W = np.roll(rolled_W, 1)
    for j in range(W.size):
      W[j] = max(abs(W[j]), abs(rolled_W[t//2 + j]))
  return W

# name = "piano" # change the name here for any of the files in the "test_samples" folder
# name = "sinusoid"
name = "tom"

'''==============='''
''' Read wav file '''
'''==============='''
W, fps = se.read_wav(f"test_samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
print(f"n={n}")
X = np.arange(n)


'''============================================================================'''
'''                                    PLOT                                    '''
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
  xaxis_title="<b>Sample <i>i</i></b>",
  yaxis_title="<b>Amplitude</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1, itemsizing='constant'),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zeroline=False)

fig.add_trace(
  go.Scattergl(
    name="Signal",
    x=X,
    y=W,
    mode="lines",
    # line_shape='spline',
    line=dict(width=1, color="black"),
  )
)

fig.add_trace(
  go.Scattergl(
    name="Envelope",
    x=X,
    y=get_reference_envelope(W),
    mode="lines",
    # line_shape='spline',
    line=dict(width=2, color="red"),
  )
)

fig.show(config=dict({'scrollZoom': True}))

# save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", "") + "_" + name + ".svg"
# fig.write_image(save_name, width=650, height=280, engine="kaleido", format="svg")
# print("saved:", save_name)