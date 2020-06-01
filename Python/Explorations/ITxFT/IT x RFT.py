import numpy as np
# from collections import deque
import plotly.graph_objects as go
# from plotly.subplots import make_subplots

#######################
#######################
n = 101
X = np.arange(n)
Y = np.zeros(n)

f = 4
p = 0
W = 2 * np.cos(p + 2 * np.pi * f * X / n)

freqs = n // 2 + 1

F = np.linspace(0, freqs - 1, freqs)
P = np.array([0, np.pi / 2])
FP = np.zeros((freqs, 2))

X, Y = [], []
for t in range(n):
  for i in range(freqs):
    for j in range(2):
      FP[i, j] += W[t] * np.cos(P[j] + 2 * np.pi * F[i] * t / n) * 2 / n

FT = np.fft.rfft(W) * 2 / n

print(np.allclose(np.array([[i.real, i.imag] for i in FT]), FP))

print(np.round(FT, 2), "\n", np.round(FP, 2))

exit()
####################################################
####################### DRAW #######################
####################################################

fig = go.Figure()
fig.update_xaxes(tickvals=P)
fig.update_yaxes(tickvals=F)

fig.add_trace(
  go.Scatter(
    name= "Amplitudes",
    x= X,
    y= Y,
    # legendgroup="Normal",
    customdata=FP.flatten('C'),
    hovertemplate='phase:%{x:.3f}<br>frequency:%{y:.3f}<br>amplitude: %{customdata:.3f} ',
    mode='markers',
    marker=dict(
        size=10,
        color=FP.flatten('C'), #set color equal to a variable
        colorscale='Bluered', # one of plotly colorscales
        showscale=False
    )
  ))

fig.add_trace(
  go.Scatter(
    name= "Predicted Amplitudes",
    x= Xs,
    y= Ys,
    # legendgroup="Normal",
    customdata=FPs.flatten('C'),
    hovertemplate='phase:%{x:.3f}<br>frequency:%{y:.3f}<br>amplitude: %{customdata:.3f} ',
    mode='markers',
    marker_symbol="circle-open",
    marker=dict(
      size=18,
      color=FPs.flatten('C'), #set color equal to a variable
      colorscale='Bluered', # one of plotly colorscales
      showscale=False,
      line=dict(
                color=FPs.flatten('C'),
                width=4
      )
    )
  ))

fig.add_trace(
  go.Scatter(
    name= "Lines",
    x= [0, 2 * np.pi, None, 0, 2 * np.pi, None, min_p, min_p, None, max_p, max_p],
    y= [local_min_f, local_min_f, None, local_max_f, local_max_f, None, 0, n, None, 0, n],
    customdata=FP.flatten('C'),
    hovertemplate='phase:%{x:.3f}<br>frequency:%{y:.3f}<br>amplitude: %{customdata:.3f} ',
    mode='lines',
    line=go.scatter.Line(color="gray")
  ))

fig.show()