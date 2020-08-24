from scipy.signal import hilbert
import numpy as np
import plotly.graph_objects as go
import numpy.polynomial.polynomial as poly
import sys

'''Generating Wave'''

np.random.seed(1)
# fps = 10
n = 1000+1

X = np.arange(n)# / fps
W = np.zeros(n)

fp = [
  [20, np.pi],
  # [2, 1.2 * np.pi],
  # [2.3, 1.4 * np.pi]
  ]

coefs = poly.polyfit([0, n//4, 3 * n//4, n], [0.6, 0.1, 0.9, .2], 3)
A = poly.polyval(X, coefs)

noise = np.random.normal(0, 0.1, n)
noise[0 : n//2] = 0

for f, p in fp:
  W += np.cos(p + 2 * np.pi * f * X / n) + noise

W = A * W / np.max(np.abs(W))

'''Analytic_Signal'''
analytic_signal = hilbert(W)
hilbert_envelope = np.abs(analytic_signal)


'''Plotting'''
fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  xaxis_title="$i$",
  yaxis_title="Amplitude",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Latin Modern",
  color="black",
  size=18
  )
)
fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=2, zerolinecolor="silver")

'''Plotting Signal'''
fig.add_trace(
  go.Scatter(
    name="Signal",
    # x=X,
    y=W,
    mode="lines",
    line=dict(
        width=.8,
        color="gray",
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="Hilbert Envelope",
    # x=X,
    y=hilbert_envelope,
    mode="lines",
    line=dict(
        width=.8,
        color="black",
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="$ i = n/2 $",
    x=[n//2, n//2],
    y=[-1, 1],
    mode="lines",
    line=dict(
        width=2,
        color="silver",
        dash="dash"
    )
  )
)

# fig.show()
save_name = "./" + sys.argv[0].split('/')[-1].replace(".py", ".pdf")
fig.write_image(save_name, width=800, height=400, scale=1, engine="kaleido")
print("saved:", save_name)
