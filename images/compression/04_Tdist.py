from plotly.subplots import make_subplots
from statistics import mode
import numpy as np
import plotly.graph_objects as go
import signal_envelope as se
from scipy import interpolate
import sys
from scipy.stats import shapiro, norm

def get_pcs(Xpc, W):
  Xpc = Xpc.astype(int)
  # amp = np.max(np.abs(W))
  # max_T = int(np.max(np.abs(Xpc[1:] - Xpc[:-1])))
  Xlocal = np.linspace(0, 1, mode(Xpc[1:] - Xpc[:-1]))

  orig_pcs = []
  norm_pcs = []
  for i in range(2, Xpc.size):
    x0 = Xpc[i - 1]
    x1 = Xpc[i] + 1
    orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1-x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      # Ylocal = Ylocal / np.max(np.abs(Ylocal)) * amp
      norm_pcs.append(Ylocal)
  return np.average(np.array(norm_pcs), 0), orig_pcs, norm_pcs

def to_plot(Matrix):
  X = []
  Y = []
  for line in Matrix:
    for x, y in enumerate(line):
      X.append(x)
      Y.append(y)
    X.append(None)
    Y.append(None)
  return X, Y

def get_periods(name):
  '''###### Read wav file ######'''
  W, fps = se.read_wav(f"./original_samples/{name}.wav")
  W = W - np.average(W)
  amp = np.max(np.abs(W))
  W = W / amp
  n = W.size

  Xpc0 = se.get_frontiers(W, mode=1)

  '''###### Read Pseudo-cycles info ######'''
  Xpc = np.genfromtxt(f"./csvs/{name}.csv", delimiter=",")

  T0 = (Xpc0[1 :] - Xpc0[: -1]).astype(np.int)
  T = (Xpc[1 :] - Xpc[: -1]).astype(np.int)
  l = max(np.max(np.abs(T0)), np.max(np.abs(T)))

  avg = mode(T0)
  sd = np.sqrt(np.average((avg - T0)**2))
  Y0 = norm.pdf(np.arange(l), avg, sd) 
  Y0 = np.max(np.bincount(T0)) * Y0 / np.max(Y0)

  avg = mode(T)
  sd = np.sqrt(np.average((avg - T)**2))
  Y = norm.pdf(np.arange(l), avg, sd)
  Y = np.max(np.bincount(T)) * Y / np.max(Y)

  return T0, T, Y0, Y

def plot(T0, T, Y0, Y, r, c):
  fig.add_trace(
    go.Scatter(
      name="Normal Approx. of Initial Dist.      ",
      y=Y0,
      fill="tozeroy",
      mode="none",
      fillcolor="rgba(9, 96, 235, .4)",
      line_shape='spline',
      # visible = "legendonly"
      showlegend=False
    ), row=r, col=c
  )

  fig.add_trace(
    go.Scatter(
      name="Normal Approx. of Final Dist.      ", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=X,
      y=Y,
      fill="tozeroy",
      mode="none",
      fillcolor="rgba(245, 7, 7 .4)",
      line_shape='spline',
      # visible = "legendonly"
      showlegend=False
    ), row=r, col=c
  )

  fig.add_trace(
    go.Scatter(
      name="Initial Periods Distribution      ", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=X,
      y=np.bincount(T0),
      # fill="toself",
      mode="lines+markers",
      line=dict(
        width=1,
        color="rgb(3, 61, 153)",
      ),
      marker=dict(
        color="rgb(3, 61, 153)",
      ),
      # visible = "legendonly"
      showlegend=False
    ), row=r, col=c
  )

  fig.add_trace(
    go.Scatter(
      name="Final Periods Distribution      ", # <|<|<|<|<|<|<|<|<|<|<|<|
      # x=X,
      y=np.bincount(T),
      # fill="toself",
      mode="lines+markers",
      line=dict(
        width=1,
        color="rgb(153, 3, 3)",
      ),
      marker=dict(
        color="rgb(153, 3, 3)",
      ),
      # visible = "legendonly"
      showlegend=False
    ), row=r, col=c
  )


'''============================================================================'''

FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

'''============================================================================'''
'''                               PLOT WAVEFORM                                '''
'''============================================================================'''
fig = make_subplots(
    rows=1, cols=4,
    subplot_titles=("cello", "piano", "sopranoA", "singerbass"),
    x_title="<b>Period</b>",
    y_title="<b>Ocurrences</b>"
)

# subplot title adjustments
fig.layout.annotations[0].update(y=0.95)
fig.layout.annotations[2].update(y=0.95)
fig.layout.annotations[1].update(y=0.95)
fig.layout.annotations[3].update(y=0.95)

fig.layout.template ="plotly_white" 
fig.update_layout(
  # xaxis_title="<b>Period</b>",
  # yaxis_title="<b>Number of ocurrences</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='center', y=1.25),
  margin=dict(r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT
fig.add_trace(
  go.Scatter(
    name="Initial Dist.", # <|<|<|<|<|<|<|<|<|<|<|<|
    y=[0],
    mode="lines+markers",
    line=dict(
      width=1,
      color="rgb(3, 61, 153)",
    ),
    marker=dict(
      color="rgb(3, 61, 153)",
    ),
  ), row=1, col=1
)
fig.add_trace(
  go.Scatter(
    name="Final Dist.", # <|<|<|<|<|<|<|<|<|<|<|<|
    y=[0],
    mode="lines+markers",
    line=dict(
      width=1,
      color="rgb(153, 3, 3)",
    ),
    marker=dict(
      color="rgb(153, 3, 3)",
    ),
  ), row=1, col=1
)
fig.add_trace(
  go.Scatter(
    name="Normal Appr.",
    y=[0],
    fill="tozeroy",
    mode="none",
    fillcolor="rgba(9, 96, 235, .4)",
    line_shape='spline',
  ), row=1, col=1
)

fig.add_trace(
  go.Scatter(
    name="Normal Appr.", # <|<|<|<|<|<|<|<|<|<|<|<|
    y=[0],
    fill="tozeroy",
    mode="none",
    fillcolor="rgba(245, 7, 7 .4)",
    line_shape='spline',
  ), row=1, col=1
)


#############################
########### cello ###########
#############################
fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black', range=[112, 351], row=1, col=1)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')

T0, T, Y0, Y = get_periods("cello")
plot(T0, T, Y0, Y, 1, 1)


#############################
########### piano ###########
#############################
fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black', range=[60, 109], row=1, col=2)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')
T0, T, Y0, Y = get_periods("piano")
plot(T0, T, Y0, Y, 1, 2)

#############################
########### sopranoA ###########
#############################
fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black', range=[19, 74], row=1, col=3)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')
T0, T, Y0, Y = get_periods("sopranoA")
plot(T0, T, Y0, Y, 1, 3)


#############################
########### singerbass ###########
#############################
fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black', range=[109, 262], row=1, col=4)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')
T0, T, Y0, Y = get_periods("singerbass")
plot(T0, T, Y0, Y, 1, 4)


fig.show(config=dict({'scrollZoom': True}))
save_name = sys.argv[0].split('/')[-1].replace(".py", "")
wid = 650
hei = 190
fig.write_image("./paper/images/" + save_name + ".svg", width=wid, height=hei, engine="kaleido", format="svg")
fig.write_image("./site/public/images/" + save_name + ".webp", width=int(1.7*wid), height=int(1.5*hei), format="webp")
fig.write_html("./site/public/images/" + save_name + ".html", include_plotlyjs="cdn", include_mathjax="cdn")
print("saved:", save_name)

