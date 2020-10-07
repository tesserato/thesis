import ctypes
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

def get_frontiers(W):
  lib = ctypes.CDLL(".\DLL.dll")

  # print("P : n in = ", W.size)

  lib.compute_raw_envelope(W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_size_t(W.size))

  pos_n = lib.get_pos_size()
  print("P : pos n = ", pos_n)
  neg_n = lib.get_neg_size()
  print("P : neg n = ", neg_n)

  lib.get_pos_X.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_size_t, shape=(pos_n,))
  lib.get_neg_X.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_size_t, shape=(neg_n,))

  pos_X = lib.get_pos_X()
  neg_X = lib.get_neg_X()

  return pos_X, neg_X

def get_raw_envelope(W):
  lib = ctypes.CDLL(".\DLL.dll")

  # print("P : n in = ", W.size)

  lib.compute_raw_envelope(W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_uint(W.size))

  pos_n = lib.get_pos_size()
  # print("P : pos n = ", pos_n)
  neg_n = lib.get_neg_size()
  # print("P : neg n = ", neg_n)

  lib.get_X.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_size_t, shape=(pos_n + neg_n,)) 

  X = lib.get_X()

  return X

# lib.frontier_from_wav("piano33.wav".encode("utf-8"))
W, _ = read_wav("brass.wav")

Xpos, Xneg = get_frontiers(W)

# print(XX)

# 180 milliseconds
# 188 milliseconds
# 155 milliseconds
# 139 milliseconds
# 122 milliseconds

# 198 milliseconds
# 110 milliseconds
# 101 milliseconds
# 92 milliseconds
# 228 milliseconds

# 156 milliseconds
# 81 milliseconds
# 94 milliseconds
# 84 milliseconds
# 96 milliseconds

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
    name="+Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=Xpos,
    y=W[Xpos],
    # fill="toself",
    mode="lines",
    line=dict(
        width=1,
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
    mode="lines",
    line=dict(
        width=1,
        color="red",
        # showscale=False
    ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))
