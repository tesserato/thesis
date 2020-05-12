# import plotly.express as px
import plotly.graph_objects as go
import numpy as np


#######################
n = 10

X = np.arange(n)

f = 3
p = np.pi / 2
W = np.cos(p + 2 * np.pi * f * X / n)
#######################

fig = go.Figure()

fig.update_xaxes(tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
fig.update_yaxes(tickvals=[i for i in range(n)])

fig.update_layout(
    # width = 2 * np.pi * 220,
    # height = n * 220,
    # yaxis = dict(scaleanchor = "x", scaleratio = 1),
    title=f"n = {n}",
    xaxis_title="Phase",
    yaxis_title="Frequency",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
px = -2*np.pi*(n - 2)
res = 20
dp = (2 * np.pi - px) / res
sum_d = np.zeros(res)
for t in range(1,n):
  # f = n
  
  fy = -n*p/(2*np.pi*t) + n/t

  for i in range(res):
    sum_d[i] += (-n*(px + i * dp)/(2*np.pi*t) + n/t - f)**2 * W[t]/n

  fig.add_trace(
    go.Scatter(
      x=[2*np.pi,px], 
      y=[0,fy],
      hoverinfo=f"all",
      name=f"t={t}, a={W[t]:.2f}", 
      mode='lines',
      line=go.scatter.Line(
        width=max(1, W[t]**2 * 10),
        dash="solid" if W[t]>0 else "dash"
        )
    )
  )

Xs = np.arange(px, 2 * np.pi, res)
fig.add_trace(
  go.Scatter(
    x=Xs,
    y=sum_d,
    name=f"sum d",
    mode='lines',
      line=go.scatter.Line(
        width=2,
        dash="solid",
        color="red"
      )
  )
)

fig.add_trace(
  go.Scatter(
    x=[p], 
    y=[f],
    name=f"max", 
    mode='markers',
    marker=dict(
        size=8,
        color="black", #set color equal to a variable
        showscale=False
    )
  )
)

fig.show(config=dict({'scrollZoom': False}))



