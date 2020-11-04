import sys
# sys.path.append("c:/Users/tesse/Desktop/Files/Dropbox/0_Thesis/Python")
import numpy as np
import plotly.graph_objects as go
# from Helper import signal_to_pulses, get_pulses_area

'''Generating Wave'''

np.random.seed(1)
# fps = 10
n = 40

X = np.arange(n)# / fps
W = np.zeros(n)

f = 1
p = np.pi / 3

W = np.cos(p + 2 * np.pi * f * X / n) + np.random.normal(0, 1 / 2, n)

FT = np.fft.rfft(W)
frequency = np.argmax(np.abs(FT))
phase = np.angle(FT[frequency])

print(f"Ori: f={f}, p={p}")
print(f"FFT: f={frequency}, p={phase}")

W = W / np.max(np.abs(W))
xmax = np.argmax(W)
xmin = np.argmin(W)

repeat = 3
W = np.tile(W, repeat)
X = np.arange(repeat * n)
Xmax = [xmax + i * n for i in range(repeat)]
Xmin = [xmin + i * n for i in range(repeat)]

'''Plotting'''
FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

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
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=2, zerolinecolor="gray")

'''Samples'''
fig.add_trace(
  go.Scatter(
    name= "Samples Stem",
    showlegend=False,
    x=[i for x in X for i in (x, x, None)],
    y=[i for y in W for i in (0, y, None)],
    mode='lines',
    line=go.scatter.Line(color="silver", width=1)
  )
)

fig.add_trace(
  go.Scatter(
    name= "Cycles",
    showlegend=False,
    x=[i for x in [j * n for j in range(1, repeat)] for i in (x + .5, x + .5, None)],
    y=[i for y in [j * n for j in range(1, repeat)] for i in (-1, 1, None)],
    mode='lines',
    line=go.scatter.Line(color="black", width=1)
  )
)

fig.add_trace(
  go.Scatter(
    name="Frames",
    x=X,
    y=W,
    mode='markers+lines',
    line=go.scatter.Line(color="gray", width=1),
    marker=dict(size=4, color="gray")
  )
)


fig.add_trace(
  go.Scatter(
    name="Maxima",
    x=Xmax,
    y=W[Xmax],
    mode='markers',
    marker_symbol="square",
    marker=dict(size=8, color="black")
  )
)


fig.add_trace(
  go.Scatter(
    name="Minima",
    x=Xmin,
    y=W[Xmin],
    mode='markers',
    marker_symbol="diamond",
    marker=dict(size=8, color="black")
  )
)

fig.show()
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=605, height=300, engine="kaleido", format="svg")
print("saved:", save_name)
