from os import linesep
import wave
import numpy as np
import plotly.graph_objects as go

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

name = "tom"
W, fps = read_wav(name + ".wav")
amp = np.max(np.abs(W))
Xpos = np.genfromtxt(name + "_pos.csv", delimiter=",").astype(int)
Xneg = np.genfromtxt(name + "_neg.csv", delimiter=",").astype(int)

Xpos_n = np.genfromtxt(name + "_pos_n.csv", delimiter=",").astype(int)
Xneg_n = np.genfromtxt(name + "_neg_n.csv", delimiter=",").astype(int)

Xpc = np.genfromtxt(name + "_Xpcs.csv", delimiter=",").astype(int)

Xpc_2 = np.genfromtxt(name + "_Xpcs_best.csv", delimiter=",").astype(int)

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
    name="Signal", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=np.arange(W.size),
    y=W,
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+Frontier n", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xpos_n,
    y=W[Xpos_n],
    # fill="toself",
    mode="markers",
    marker=dict(
        size=4,
        color="blue",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="-Frontier n", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xneg_n,
    y=W[Xneg_n],
    # fill="toself",
    mode="markers",
    marker=dict(
        size=4,
        color="blue",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="+Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xpos,
    y=W[Xpos],
    # fill="toself",
    mode="markers",
    marker=dict(
        size=4,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="-Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xneg,
    y=W[Xneg],
    # fill="toself",
    mode="markers",
    marker=dict(
        size=4,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

XX = []
YY = []
for x in Xpc:
  XX.append(x), XX.append(x), XX.append(None)
  YY.append(-amp), YY.append(amp), YY.append(None)
fig.add_trace(
  go.Scatter(
    name="Xpc", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=XX,
    y=YY,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="black",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

XX = []
YY = []
for x in Xpc_2:
  XX.append(x), XX.append(x), XX.append(None)
  YY.append(-amp), YY.append(amp), YY.append(None)
fig.add_trace(
  go.Scatter(
    name="Xpc_2", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=XX,
    y=YY,
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
        color="green",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)
fig.show(config=dict({'scrollZoom': True}))
