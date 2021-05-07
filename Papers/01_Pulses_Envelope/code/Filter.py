import plotly.graph_objects as go
import signal_envelope as se
import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, fps, cutoff = 2000, order = 2):
  nyq = 0.5 * fps
  normal_cutoff = cutoff / nyq
  # Get the filter coefficients 
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = filtfilt(b, a, data)
  return y

'''==============='''
''' Read wav file '''
'''==============='''
name = "alto"
W, fps = se.read_wav(f"C:/Users/tesse/Desktop/Files/Dropbox/0_Science/reps/envelope/test_samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size
print(f"n={n}")
X = np.arange(n)



y = butter_lowpass_filter(W, fps)

fig = go.Figure()
fig.add_trace(go.Scatter(
            y = W,
            # line =  dict(shape =  'spline' ),
            name = 'signal with noise'
            ))
fig.add_trace(go.Scatter(
            y = y,
            # line =  dict(shape =  'spline' ),
            name = 'filtered signal'
            ))
fig.show()