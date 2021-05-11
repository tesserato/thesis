import sys
print(sys.version)
import plotly.graph_objects as go
sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python/01_Envelope")
import numpy as np
from scipy.interpolate import interp1d
from Helper import signal_to_pulses, get_pulses_area
from plotly.subplots import make_subplots

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
for p in pulses[1:]:
  Xp.append(p.x)
  Yp.append(np.abs(p.y))

Xp = np.array(Xp)
Yp = np.array(Yp)


scaling = (np.sum(Xp[1:] - Xp[:-1])) / np.sum(Yp)
Y = Yp * scaling


'''============================================================================'''
'''                              PLOT LINES                                    '''
'''============================================================================'''
fig = go.Figure()

FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

for i in fig['layout']['annotations']:
    i['font'] = FONT
    
'''Plotting'''

fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="<b><i>x</i></b>", # ( <b><i>i</i></b> ) ",
  yaxis_title="<b><i>y</i></b>", # ( <b><i>i</i></b> ) ",
  # yaxis = dict(scaleanchor = "x", scaleratio = 1),
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zeroline=False)


fig.add_trace(
  go.Scatter(
    name= "Stem",
    showlegend=False,
    x=[i for x in Xp for i in (x, x, None)],
    y=[i for y in Y for i in (0, y, None)],
    mode='lines',
    line=dict(color="silver", width=1)
  )
)

fig.add_trace(
  go.Scatter(
    name="P      ",
    # showlegend=False,
    x=Xp,
    y=Y,
    mode='markers',
    marker=dict(size=7, color="black")
  )
)

for i in range(1, len(Xp)):    
  fig.add_annotation(
    x=Xp[i],  # arrows' head
    y=Y[i],  # arrows' head
    ax=Xp[i - 1],  # arrows' tail
    ay=Y[i - 1],  # arrows' tail
    xref='x',
    yref='y',
    axref='x',
    ayref='y',
    text='',  # if you want only the arrow
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor='gray'
  )



fig.add_trace(
  go.Scatter(
    name="V      ",
    # showlegend=False,
    x=[None],
    y=[None],
    mode='lines',
    # fill="tozeroy",
    # fillcolor="silver",
    line=dict(color="gray", width=1),
    marker=dict(size=7, color="black")
  )
)


fig.add_annotation(
  x=0,  # arrows' head
  y=8,  # arrows' head
  ax=0,  # arrows' tail
  ay=-1,  # arrows' tail
  xref='x',
  yref='y',
  axref='x',
  ayref='y',
  text='',  # if you want only the arrow
  showarrow=True,
  arrowhead=2,
  arrowsize=1,
  arrowwidth=2,
  arrowcolor='black'
)


fig.add_annotation(
  x=Xp[-1] + 2,  # arrows' head
  y=0,  # arrows' head
  ax=-1,  # arrows' tail
  ay=0,  # arrows' tail
  xref='x',
  yref='y',
  axref='x',
  ayref='y',
  text='',  # if you want only the arrow
  showarrow=True,
  arrowhead=2,
  arrowsize=1,
  arrowwidth=2,
  arrowcolor='black'
)

fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=200, engine="kaleido", format="svg")
print("saved:", save_name)
