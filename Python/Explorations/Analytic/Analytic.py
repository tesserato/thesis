import numpy as np
import plotly.graph_objects as go

#######################
#######################
n = 100
X = np.arange(n)
f = 10
p = 2
W = 2 * np.cos(p + 2 * np.pi * f * X / n)

freqs = n // 2 + 1

P = np.zeros(freqs)

S, C = 0, 0
for f_idx in range(freqs):
  for x in range(n):
      # S1 += W[x] * np.sin(2 * np.pi * f_idx * x / n)
      # S2 += -2 * W[x] * np.sin(np.pi * f_idx * x / n)**2
      # S3 += W[x]
      # P[f_idx] = - np.arctan(S1 / (S2 + S3))
      S += W[x] * np.sin(2 * np.pi * f_idx * x / n)
      C += W[x] * np.cos(2 * np.pi * f_idx * x / n)
      # res = S / C
      P[f_idx] = -np.arctan(S / C)

FT = np.fft.rfft(W) * 2 / n

print(f, p, ">>", np.round(P[f], 4))


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