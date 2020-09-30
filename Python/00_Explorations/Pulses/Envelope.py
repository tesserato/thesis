import plotly.graph_objects as go
import random
import numpy as np
from Helper import read_wav
from scipy.signal import savgol_filter

class Pulse:
  # pulses = []
  # max_length = 0
  def __init__(self, x0, W):
    # Pulse.pulses.append(self)
    self.start = x0
    self.end = self.start + W.shape[0]
    # self.n = self.end - self.start
    # self.W = W
    # self.normalized_W = W / np.max(np.abs(W))
    self.avg = np.average(W)
    # self.var = np.var(W)
    # centered_W = W - np.average(W)
    # FT = np.fft.rfft(centered_W)
    # self.f = np.argmax(np.abs(FT))
    # self.p = np.angle(FT[self.f])
    # sign = np.sign(W[0])
    # self.zeros = 0
    # for w in centered_W:
      # if sign != np.sign(w):
        # self.zeros += 1
        # sign = np.sign(w)

def refine_pulses(pulses, TH):
  refined_pulses = []
  allgaps = []
  gaps = []
  for i in range(len(pulses)):
    if np.abs(pulses[i].avg) >= TH[i]:
      if len(gaps) > 0:
        # print(gaps)
        allgaps.append((len(refined_pulses) - 1, gaps))
      gaps = []
      refined_pulses.append(pulses[i])
    else:
      gaps.append(i)

  # TODO improve this part
  for gaps in allgaps:
    idx, actual_gaps = gaps
    if len(actual_gaps) <=1:
      if np.abs(refined_pulses[idx].avg) - TH[actual_gaps[0]-1] <= np.abs(refined_pulses[idx+1].avg) - TH[actual_gaps[0]+1]:
        start = pulses[actual_gaps[0]-1].start
        end = pulses[actual_gaps[0]].end
        refined_pulses[idx] = Pulse(start, W[start : end])
      else:
        start = pulses[actual_gaps[0]].start
        end = pulses[actual_gaps[0] + 1].end
        refined_pulses[idx + 1] = Pulse(start, W[start : end])
    else:
      if len(actual_gaps) % 2 == 0:
        start = pulses[actual_gaps[0]-1].start
        end = pulses[actual_gaps[len(actual_gaps) // 2 - 1]].end
        refined_pulses[idx] = Pulse(start, W[start : end])

        start = pulses[actual_gaps[len(actual_gaps) // 2]].start
        end = pulses[actual_gaps[-1] + 1].end
        refined_pulses[idx + 1] = Pulse(start, W[start : end])
      else:
        if np.abs(refined_pulses[idx].avg) - TH[actual_gaps[0]-1] <= np.abs(refined_pulses[idx+1].avg) - TH[actual_gaps[-1]+1]:
          middle = len(actual_gaps) // 2
        else:
          middle = len(actual_gaps) // 2 - 1

        start = pulses[actual_gaps[0]-1].start
        end = pulses[actual_gaps[middle]].end
        refined_pulses[idx] = Pulse(start, W[start : end])

        start = pulses[actual_gaps[middle + 1]].start
        end = pulses[actual_gaps[-1] + 1].end
        refined_pulses[idx + 1] = Pulse(start, W[start : end])
  return refined_pulses

W, fps = read_wav("Samples/piano33.wav")
W = W - np.average(W)
a = np.max(np.abs(W))
W = W / a

# W = savgol_filter(W, 5, 3)

# W = W [ : W.size // 4]

n = W.shape[0]
X = np.arange(n)

## Split the signal into pulses
pulses = []
sign = np.sign(W[0])
x0 = 0
max_length = 0
for x in range(n):
  if sign != np.sign(W[x]):
    # w = W[x0 : x]
    pulses.append(Pulse(x0, W[x0 : x]))
    if x - x0 > max_length:
      max_length = x - x0
    x0 = x
    sign = np.sign(W[x])

pulses.append(Pulse(x0, W[x0 : n]))
if n - x0 > max_length:
  max_length = n - x0

print(f"Max length of a pulse={max_length}")

## Plot original signal, pulses and amplitude
XX = []
YY = []
XX_avg = []
YY_avg = []
for p in pulses:
  XX.append(p.start - .5)
  XX.append(p.end - .5)
  XX.append(None)
  YY.append(p.avg)
  YY.append(p.avg)
  YY.append(None)
  XX_avg.append((p.start + p.end) / 2 - .5)
  YY_avg.append(np.abs(p.avg))

fig = go.Figure()
fig.layout.template ="plotly_white"

fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

fig.update_layout(
  xaxis_title="x",
  yaxis_title="Amplitude",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=5, r=5, b=5, t=5),
  font=dict(
  family="Computer Modern",
  color="black",
  size=18
  )
)

fig.add_trace(
  go.Scatter(
    x=X,
    y=W,
    name="Signal",
    mode="lines",
    line=dict(
        # size=8,
        color="silver",
        # showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    x=XX,
    y=YY,
    fill="tozeroy",
    name="Pulses",
    mode="none",
    fillcolor="rgba(0,0,0,0.16)"
  )
)

fig.add_trace(
  go.Scatter(
    x=XX_avg,
    y=YY_avg,
    name="Pulse Amplitude",
    mode="markers",
    marker=dict(
        size=10,
        color="black",
        showscale=False
    )
  )
)

coefs = np.polyfit(XX_avg, YY_avg, 4)
YY_fit = np.polyval(coefs, XX_avg)

fig.add_trace(
  go.Scatter(
    x=XX_avg,
    y=YY_fit,
    name="Polynomial Fit",
    mode="lines",
    line=dict(
      width=2,
      color="gray"
    )
  )
)

# fig.add_trace(
#   go.Scatter(
#     x=XX_avg,
#     y=YY_fit + np.abs(YY_avg - YY_fit),
#     name="Deviations from Polynomial Fit",
#     mode="markers",
#     # marker=dict(
#     #     size=8,
#     #     color="red", #set color equal to a variable
#     #     showscale=False
#     # )
#   )
# )

# coefs = np.polyfit(XX_avg, np.abs(YY_avg - YY_fit), 0)
# YY_std = np.polyval(coefs, XX_avg)

# fig.add_trace(
#   go.Scatter(
#     x=XX_avg,
#     y=YY_fit + YY_std,
#     name="Average + Standard Deviation",
#     mode="lines",
#     line=dict(
#         color="gray",
#         dash="dot" # 'solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'
#     )
#   )
# )

# TH = YY_fit# - YY_std

# fig.add_trace(
#   go.Scatter(
#     x=XX_avg,
#     y=TH,
#     name="Average - Standard Deviation",
#     mode="lines",
#     line=dict(
#         color="gray",
#         dash="dash"
#     )
#   )
# )

##########################################################
##########################################################
refined_pulses = refine_pulses(pulses, YY_fit)


print(len(pulses), len(refined_pulses))

XX = []
YY = []
XX_avg = []
YY_avg = []
for p in refined_pulses:
  XX.append(p.start - .5)
  YY.append(0)

  XX.append(p.start - .5)
  YY.append(p.avg)

  XX.append(p.end - .5)
  YY.append(p.avg)

  XX.append(p.end - .5)
  YY.append(0)

  XX.append(None)
  YY.append(None)

  XX_avg.append((p.start + p.end) / 2 - .5)
  YY_avg.append(np.abs(p.avg))

fig.add_trace(
  go.Scatter(
    x=XX_avg,
    y=YY_avg,
    name="Refined Pulse Amplitude",
    mode="markers",
    marker=dict(
        size=4,
        color="red",
        showscale=False
    )
  )
)

fig.add_trace(
  go.Scatter(
    x=XX,
    y=YY,
    fill="tozeroy",
    name="Refined Pulses",
    mode="none",
    fillcolor="rgba(100,0,0,0.16)"
  )
)


fig.show(config=dict({'scrollZoom': True}))
# fig.write_image("./01ContinuousVsDiscrete.pdf", width=800, height=400, scale=1)

