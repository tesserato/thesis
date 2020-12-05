import sys
print(sys.version)
import plotly.graph_objects as go
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/01_Envelope")
import numpy as np
from scipy.interpolate import interp1d
from Helper import signal_to_pulses, get_pulses_area

'''Generating Wave'''

np.random.seed(0)
# fps = 10
n = 100

X = np.arange(n)# / fps
C = np.zeros(n)

f = interp1d([0, 0.25 * n, 0.5 * n, 0.75 * n, n], np.abs(1 + np.random.normal(0, .3, 5)), "cubic")
E = f(X)

afp = [
  [1, 5, 1.4 * np.pi],
  [1.5, 10, 1.4 * np.pi],
  # [1, 2 * 6, 1.4 * np.pi]
  ]
for a, f, p in afp:
  C += a * np.cos(p + 2 * np.pi * f * X / n) #+ np.random.normal(0, a / 100, n)

C = -C / np.max(np.abs(C))

W = E * C

'''Wave to pulses'''
pulses = signal_to_pulses(C)
pulses_X, pulses_Y = get_pulses_area(pulses)

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
fig.update_yaxes(showline=False, zeroline=False, showgrid=True, gridwidth=1, gridcolor='silver', tickvals=[-1, 0, 1])

fig.add_trace(
  go.Scatter(
    name="Envelope <b>e</b>      ",
    x=X,#pulses_X,
    y=E,#pulses_Y,
    mode="lines",
    line=dict(width=2, color="gray", shape = 'spline'),
    marker=dict(size=3, color="gray")
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Carrier <b>c</b>      ",
    # showlegend=False,
    x=X,
    y=C,
    mode='lines+markers',
    fill="tozeroy",
    fillcolor="silver",
    line=dict(width=1, color="gray"),
    marker=dict(size=4, color="gray")
  )
)

fig.add_trace(
  go.Scatter(
    name="Wave <b>w = e</b> ⊙ <b>c</b>      ",
    x=X,#pulses_X,
    y=W,#pulses_Y,
    mode="lines+markers",
    line=dict(width=1, color="black"),
    marker=dict(size=4, color="black")
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Local Extrema      ",
    x=[p.x for p in pulses][1:],
    y=W[[p.x for p in pulses]][1:],
    mode="markers",
    marker_symbol="circle-open",
    # line=dict(width=1, color="black"),
    marker=dict(size=10, color="black")
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=350, engine="kaleido", format="svg")
print("saved:", save_name)
