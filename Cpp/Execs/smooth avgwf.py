from os import linesep
import wave
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
# from numpy.polynomial import polynomial

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

name = "bend"
avgwf = np.genfromtxt(name + "_avgpcw_best.csv", delimiter=",")

for i in range(avgwf.size):
  avgwf[i] = i / avgwf.size
# avgwf = avgwf[::-1]

avgwf_old = np.copy(avgwf)


dmax = np.abs(np.max(avgwf[1:]-avgwf[:-1]))
dmin = -np.abs(np.min(avgwf[1:]-avgwf[:-1]))


avg = (avgwf[0] + avgwf[-1]) / 2

dev = avgwf[0] - avgwf[-1]
if dev > dmax:
  print("start maior")
  avgwf[0]  = avg + dmax / 2
  avgwf[-1] = avg - dmax / 2
elif dev < dmin:
  print("start menor")
  avgwf[0]  = avg + dmin / 2
  avgwf[-1] = avg - dmin / 2

no2 = avgwf.size//2
for i in range(avgwf.size//2):
  dev = avgwf[i + 1] - avgwf[i]
  if dev > dmax:
    print(i, "maior")
    avgwf[i + 1] = ((avgwf[i] + dmax) * (no2 - i) + avgwf[i + 1] * i) / no2
  elif dev < dmin:
    print(i, "menor")
    avgwf[i + 1] = ((avgwf[i] + dmin) * (no2 - i) + avgwf[i + 1] * i) / no2

for i in range(avgwf.size//2):
  dev = avgwf[avgwf.size - i - 2] - avgwf[avgwf.size - i - 1]
  if dev > dmax:
    print(i, "maior")
    avgwf[avgwf.size - i - 2] = ((avgwf[avgwf.size - i - 1] + dmax) * (no2 - i) + avgwf[avgwf.size - i - 2] * i) / no2
  elif dev < dmin:
    print(i, "menor")
    avgwf[avgwf.size - i - 2] = ((avgwf[avgwf.size - i - 1] + dmin) * (no2 - i) + avgwf[avgwf.size - i - 2] * i) / no2


'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="x",
  yaxis_title="Amplitude",

  # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
  # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

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
    name="avgwf_old", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=np.arange(W.size),
    y=avgwf_old,
    mode="markers",
    line=dict(
        # size=8,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="avgwf_old shifted", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=np.arange(avgwf.size) + avgwf.size,
    y=avgwf_old,
    mode="markers",
    line=dict(
        # size=8,
        color="blue",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="avgwf", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=np.arange(W.size),
    y=avgwf,
    mode="markers",
    line=dict(
        # size=8,
        color="green",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="avgwf shifted", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=np.arange(avgwf.size) + avgwf.size,
    y=avgwf,
    mode="markers",
    line=dict(
        # size=8,
        color="green",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="fit", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=XX - XX.size//2 + avgwf.size,
#     y=fitted,
#     mode="lines+markers",
#     line=dict(
#         # size=8,
#         color="green",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )


fig.add_trace(
  go.Scatter(
    name="average line", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=[0, 2 * avgwf_old.size],
    y=[avg, avg],
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

# fig.add_trace(
#   go.Scatter(
#     name="avgwf new shifted", # <|<|<|<|<|<|<|<|<|<|<|<|
#     x=np.arange(avgwf.size) + avgwf.size,
#     y=avgwf_new,
#     mode="markers",
#     line=dict(
#         # size=8,
#         color="green",
#         # showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )




fig.show(config=dict({'scrollZoom': True}))
