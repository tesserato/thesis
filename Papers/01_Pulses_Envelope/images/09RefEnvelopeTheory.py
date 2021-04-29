import sys
print(sys.version)
import plotly.graph_objects as go
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/01_Envelope")
import numpy as np
from scipy.interpolate import interp1d
from Helper import signal_to_pulses, get_pulses_area

'''Generating Wave'''

def get_W(X, E, p=0):
  C = np.zeros(E.size)
  afp = [
    [1, 5, p],
    # [1.5, 10, p],
    ]
  for a, f, p in afp:
    C += a * np.cos(p + 2 * np.pi * f * X / n) #+ np.random.normal(0, a / 100, n)
  C = C / np.max(np.abs(C))
  return E * C

np.random.seed(0)
# fps = 10
n = 100
X = np.arange(n)# / fps
f = interp1d(np.linspace(0, n, 7, endpoint=True), np.abs(1 + np.random.normal(0, .3, 7)), "cubic")
E = f(X)







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
  xaxis_title="<b>Sample <i>i</i></b>",
  yaxis_title="<b>Amplitude</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, zeroline=False, showgrid=True)

fig.add_trace(
  go.Scatter(
    name="Envelope      ",
    x=X,#pulses_X,
    y=E,#pulses_Y,
    mode="lines",
    line=dict(width=2, color="red", shape = 'spline'),
    marker=dict(size=3, color="red")
    # visible = "legendonly"
  )
)

Xf = []
Yf = []
for p in np.linspace(0, 2 * np.pi, 5):
  Yf += get_W(X, E, p).tolist() + [None]
  Xf += np.arange(X.size).tolist() + [None]

fig.add_trace(
  go.Scatter(
    name="Family of sinusoids with same frequency and amplitude      ",
    x=Xf,#pulses_X,
    y=Yf,#pulses_Y,
    mode="lines",
    line=dict(width=1, color="gray", shape = 'spline'),
    marker=dict(size=4, color="gray"),
    # showlegend=False
    # visible = "legendonly"
  )
)


fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=350, engine="kaleido", format="svg")
print("saved:", save_name)
