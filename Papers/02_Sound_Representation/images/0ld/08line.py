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

f = 1
p = np.pi / 3

W = np.cos(p + 2 * np.pi * f * X / n) + np.random.normal(0, 1 / 2, n)
W = W / np.max(np.abs(W))

xmax = np.argmax(W)
xmin = np.argmin(W)
repeat = 3
W = np.tile(W, repeat)
X = np.arange(repeat * n)
Xmax = [xmax + i * n for i in range(repeat)]
Xmin = [xmin + i * n for i in range(repeat)]

FT = np.fft.rfft(W)
frequency = np.argmax(np.abs(FT))
phase = np.angle(FT[frequency])

print(f"Ori: f={f}, p={p}")
print(f"FFT: f={frequency}, p={phase}")
print(f"POS: f={n * repeat * (len(Xmax) - 1) / (Xmax[-1] - Xmax[0])}, p={2 * np.pi * Xmax[0] / n}")

# exit()

'''Plotting'''

FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )

fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="<b>Ordinality</b>",
  yaxis_title="<b><i>i</i></b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)

# fig.update_xaxes(
#     # ticktext=["End of Q1", "End of Q2", "End of Q3", "End of Q4"],
#     tickvals=[i for i in range(repeat)],
# )

fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT
fig.update_xaxes(showline=False, showgrid=True, zerolinewidth=2, zerolinecolor="gray", gridwidth=1, gridcolor='silver', tickvals=[i for i in range(repeat)])
fig.update_yaxes(showline=False, showgrid=True, zerolinewidth=2, zerolinecolor="gray", gridwidth=1, gridcolor='silver', tickvals=[i for i in Xmax] + [i for i in Xmin]
, range=[0, 120])

'''Samples'''

fig.add_trace(
  go.Scatter(
    name="Maxima",
    # x=X,
    y=Xmax,
    mode='markers+lines',
    line=go.scatter.Line(color="black", width=1),
    marker_symbol="square",
    marker=dict(size=10, color="black")
  )
)

fig.add_trace(
  go.Scatter(
    name="Minima",
    # x=X,
    y=Xmin,
    mode='markers+lines',
    line=go.scatter.Line(color="black", width=1),
    marker_symbol="diamond",
    marker=dict(size=10, color="black")
  )
)

fig.show()
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=605, height=300, engine="kaleido", format="svg")
print("saved:", save_name)
