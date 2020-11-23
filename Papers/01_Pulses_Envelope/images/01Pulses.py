import sys
print(sys.version)
import plotly.graph_objects as go
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/01_Envelope")
import numpy as np

from Helper import signal_to_pulses, get_pulses_area

'''Generating Wave'''

np.random.seed(1)
# fps = 10
n = 100+1

X = np.arange(n)# / fps
W = np.zeros(n)

afp = [
  [2, 3, np.pi],
  # [1, 2, 1.2 * np.pi],
  # [1, 2.3, 1.4 * np.pi]
  ]
for a, f, p in afp:
  W += a * np.cos(p + 2 * np.pi * f * X / n) + np.random.normal(0, a / 5, n)

W = W / np.max(np.abs(W))

'''Wave to pulses'''
pulses = signal_to_pulses(W)
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
  yaxis_title="<b>Normalized Amplitude</b>",
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
    mode='markers',
    fill="tozeroy",
    fillcolor="gray",
    marker=dict(
        size=5,
        color="black",
    )
  )
)
fig.add_trace(
  go.Scatter(
    name= "Samples Stem",
    showlegend=False,
    x=[i for x in X for i in (x, x, None)],
    y=[i for y in W for i in (0, y, None)],
    mode='lines',
    line=go.scatter.Line(color="black", width=1)
  )
)

''' Samples Legend'''
fig.add_trace(
  go.Scatter(
    name="Samples",
    x=[None],#pulses_X,
    y=[None],#pulses_Y,
    # fill="tozeroy",
    mode="markers",
      marker=dict(
      size=5,
      color="black",
    )
    # visible = "legendonly"
  )
)

''' Pulses Legend'''
fig.add_trace(
  go.Scatter(
    name="Half Pseudo Cycles",
    x=[None],#pulses_X,
    y=[None],#pulses_Y,
    fill="tozeroy",
    mode="none",
    fillcolor="gray",
    marker=dict(
      size=5,
      color="black",
    )
    # visible = "legendonly"
  )
)

''' Pseudo-Cycles '''
cycles_X = []
cycles_Y = []
for x in pulses_X[2:-3:4]:
  cycles_X.append(x)   , cycles_Y.append(-1)
  cycles_X.append(x)   , cycles_Y.append(1)
  cycles_X.append(None), cycles_Y.append(None)

fig.add_trace(
  go.Scatter(
    name="Divisions between Pseudo Cycles  ",
    x=cycles_X,
    y=cycles_Y,
    # marker_symbol="line-ns",
    mode="lines",
    line=dict(
      # width=5,
      color="black",
      dash="dot"
    )
    # marker_line_width=2, 
    # marker_size=50,
  )
)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
print("saved:", save_name)
