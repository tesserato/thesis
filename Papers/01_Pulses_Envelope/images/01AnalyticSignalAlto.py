from scipy.signal import hilbert
import numpy as np
import plotly.graph_objects as go
import signal_envelope as se
import sys
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, fps, cutoff = 10, order = 2):
  nyq = 0.5 * fps
  normal_cutoff = cutoff / nyq
  # Get the filter coefficients 
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = filtfilt(b, a, data)
  return y

name = "alto"

'''==============='''
''' Read wav file '''
'''==============='''
W, fps = se.read_wav(f"test_samples/{name}.wav")
W = W - np.average(W)
amplitude = np.max(np.abs(W))
W = W / amplitude
n = W.size

'''Analytic_Signal'''
analytic_signal = hilbert(W)
hilbert_envelope = np.abs(analytic_signal)
hilbert_envelope_filtered = butter_lowpass_filter(hilbert_envelope, fps, 100)


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
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1, itemsizing='constant'),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  titlefont=FONT
)
fig.layout.xaxis.title.font=FONT
fig.layout.yaxis.title.font=FONT
fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=.8, zerolinecolor="silver")

'''Plotting Signal'''
fig.add_trace(
  go.Scatter(
    name="Signal   ",
    # x=X,
    y=W,
    mode="lines",
    marker=dict(size=2,),
    line=dict(
        width=1,
        color="gray",
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="Hilbert Transform   ",
    # x=X,
    y=hilbert_envelope,
    mode="lines",
    line_shape='spline',
    # fillcolor="rgba(250, 140, 132, 0.5)",
    # fill="toself",
    line=dict(
        width=1,
        color="rgba(250, 0, 0, 0.5)",
    )
  )
)

fig.add_trace(
  go.Scatter(
    name="Filtered Hilbert Transform   ",
    # x=X,
    y=hilbert_envelope_filtered,
    mode="lines",
    line_shape='spline',
    line=dict(
        width=1,
        color="black",
    )
  )
)


fig.show(config=dict({'scrollZoom': True}))
save_name = "./images/" + sys.argv[0].split('/')[-1].replace(".py", ".svg")
fig.write_image(save_name, width=650, height=300, engine="kaleido", format="svg")
print("saved:", save_name)
