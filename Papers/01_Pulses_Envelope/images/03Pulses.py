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

f = interp1d([0, 0.25 * n, 0.5 * n, 0.75 * n, n], np.abs(np.random.normal(0, .3, 5)), "cubic")
E = 1+f(X)

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
pulses = signal_to_pulses(W)

Xp = []
Yp = []
X_neg_stem = []
Y_neg_stem = []
for p in pulses[1:-1]:
  Xp.append(p.x)
  Yp.append(p.y)
  if p.y < 0:
    X_neg_stem.append(p.x), X_neg_stem.append(p.x), X_neg_stem.append(None)
    Y_neg_stem.append(0), Y_neg_stem.append(p.y), Y_neg_stem.append(None)

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
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=2, zerolinecolor='black')

'''Samples'''
fig.add_trace(
  go.Scatter(
    name="Samples",
    showlegend=False,
    x=X,
    y=W,
    mode='lines+markers',
    fill="tozeroy",
    fillcolor="rgba(0,0,0,0.2)",
    line=dict(color="gray", width=1),
    marker=dict(size=5, color="gray")
  )
)

''' Samples Legend'''
fig.add_trace(
  go.Scatter(
    name="Samples      ",
    x=[None],#pulses_X,
    y=[None],#pulses_Y,
    mode="markers",
    marker=dict(size=5, color="gray")
    # visible = "legendonly"
  )
)

''' Pulses Legend'''
fig.add_trace(
  go.Scatter(
    name="Pulses      ",
    x=[None],#pulses_X,
    y=[None],#pulses_Y,
    fill="tozeroy",
    mode="none",
    fillcolor="silver",
    # marker=dict(
    #   size=5,
    #   color="black",
    # )
    # visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="Local Extrema      ",
    # showlegend=False,
    x=Xp,
    y=Yp,
    mode='markers',
    marker_symbol="circle-open",
    # fill="tozeroy",
    # fillcolor="silver",
    # line=dict(color="gray", width=1),
    marker=dict(size=12, color="darkslategray", line=dict(width=.5))
  )
)

fig.add_trace(
  go.Scatter(
    name="Absolute Local Extrema (<i>P</i>)      ",
    # showlegend=False,
    x=Xp,
    y=np.abs(Yp),
    mode='markers',
    # fill="tozeroy",
    # fillcolor="silver",
    # line=dict(color="gray", width=1),
    marker=dict(size=7, color="black")
  )
)

fig.add_trace(
  go.Scatter(
    name= "Samples Stem",
    showlegend=False,
    x=[i for x in Xp for i in (x, x, None)],
    y=[i for y in Yp for i in (0, abs(y), None)],
    mode='lines',
    line=dict(color="black", width=2)
  )
)

fig.add_trace(
  go.Scatter(
    name= "Samples negative Stem",
    showlegend=False,
    x=X_neg_stem,
    y=Y_neg_stem,
    mode='lines',
    line=dict(color="black", width=1, dash="dot")
  )
)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
print("saved:", save_name)
