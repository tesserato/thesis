import sys
import signal_envelope as se

print(sys.version)
import plotly.graph_objects as go
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/01_Envelope")
import numpy as np
from scipy.interpolate import interp1d
from Helper import signal_to_pulses, get_pulses_area

name = "nonperiodic"

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

Ex = se.get_frontiers(W, 1)
f = interp1d(Ex, np.abs(W[Ex]), kind="linear", fill_value="extrapolate", assume_sorted=False)
envY = f(X)

# C = (W / envY)[200:]
# W = W[200:]
# ftc = np.abs(np.fft.rfft(C))


ftw = np.fft.rfft(W, 2 * n)


'''============================================================================'''
'''                              PLOT LINES                                    '''
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
  xaxis_title="<b>Frequency</b>",
  yaxis_title="<b>Power</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)#, range=[820, 910])
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver')
# fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='silver')

# fig.add_trace(
#   go.Scatter(
#     name="Envelope <b>e</b>      ",
#     x=X,#pulses_X,
#     y=E,#pulses_Y,
#     mode="lines",
#     line=dict(width=2, color="gray", shape = 'spline'),
#     marker=dict(size=3, color="gray")
#     # visible = "legendonly"
#   )
# )


fig.add_trace(
  go.Scatter(
    name="Real      ",
    # x=Xw,
    y=[np.real(f) for f in ftw],
    mode="none",
    fill="tozeroy",
    fillcolor="black",
    line=dict(width=1, color="black"),
    # marker=dict(size=5, color="black")
  )
)

fig.add_trace(
  go.Scatter(
    name="Imaginary      ",
    # x=Xc,
    y=[np.imag(f) for f in ftw],
    mode='none',
    fill="tozeroy",
    fillcolor="rgba(250, 140, 132, 0.5)",
    line=dict(width=1, color="silver"),
    # marker=dict(size=4, color="gray")
  )
)

fig.show(config=dict({'scrollZoom': True}))
# save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
# fig.write_image(save_name, width=650, height=150, engine="kaleido", format="svg")
# print("saved:", save_name)
