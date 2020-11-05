import plotly.graph_objects as go
import numpy as np
import signal_envelope as se
from plotly.subplots import make_subplots
import sys
import os
os.chdir("../..")
print (os.path.abspath(os.curdir))

'''==============='''
''' Read wav file '''
'''==============='''

name = "piano33"
W, fps = se.read_wav(f"./Python/Samples/{name}.wav")
W = W - np.average(W)
n = W.size
X = np.arange(n)

save_path = os.path.abspath(os.curdir) + '''\\Papers\\02_Sound_Representation\\images\\''' + sys.argv[0].split('/')[-1].replace(".py", "") + "-" + name + ".svg"
print(save_path)


Xpos, Xneg = se.get_frontiers(W)

posL = []
for i in range(1, Xpos.size):
  posL.append(Xpos[i] - Xpos[i - 1])

negL = []
for i in range(1, Xneg.size):
  negL.append(Xneg[i] - Xneg[i - 1])

np.histogram(Xneg, np.arange(Xneg.size))


'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
fig.layout.template ="plotly_white"
# fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_layout(
  # title = name,
  xaxis_title="Length",
  yaxis_title="Number of Ocurrences",

  # yaxis = dict(        # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleanchor = "x", # <|<|<|<|<|<|<|<|<|<|<|<|
  #   scaleratio = 1,    # <|<|<|<|<|<|<|<|<|<|<|<|
  # ),                   # <|<|<|<|<|<|<|<|<|<|<|<|

  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
    family="Latin Modern Roman",
    color="black",
    size=10
  )
)

fig.add_trace(
  go.Histogram(
    name=f"Lengths of Positive Pseudo-Pulses (std={np.round(np.std(posL), 2)})",
    x=posL,
    marker_color = "black",
    xbins=dict( # bins used for histogram
      # start=-4.0,
      # end=3.0,
      size=1
    )
  ),row=1,col=1
)

fig.add_trace(
  go.Histogram(
    name=f"Lengths of Negative Pseudo-Pulses (std={np.round(np.std(negL), 2)})",
    x=negL,
    marker_color = "gray",
    xbins=dict( # bins used for histogram
      # start=-4.0,
      # end=3.0,
      size=1
    )
  ),row=1,col=2
)

# fig.add_trace(
#   go.Scatter(
#     name="-Frontier", # <|<|<|<|<|<|<|<|<|<|<|<|
#     # x=Xpos,
#     y=negL,
#     # fill="toself",
#     mode="markers",
#     marker=dict(
#         size=3,
#         color="red",
#         showscale=False
#     ),
#     # visible = "legendonly"
#   )
# )

# fig.show(config=dict({'scrollZoom': True}))
fig.write_image(save_path, width=680, height=200, scale=1, engine="kaleido", format="svg")