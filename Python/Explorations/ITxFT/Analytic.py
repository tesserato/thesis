import numpy as np
import plotly.graph_objects as go

#######################
#######################
n = 10
X = np.arange(n)
f = 2
p = 6
W = 2 * np.cos(p + 2 * np.pi * f * X / n)

freqs = n // 2 + 1

P = np.zeros(freqs)
I = np.zeros((freqs, 2))

for f_idx in range(freqs):
  S, C = 0, 0
  for x in range(n):
    S -= W[x] * np.sin(2 * np.pi * f_idx * x / n)
    C += W[x] * np.cos(2 * np.pi * f_idx * x / n)
    P[f_idx] = -np.arctan(S / C) + (0 if -np.arctan(S / C) >= 0 else np.pi)
  I[f_idx, 0] = C
  I[f_idx, 1] = S

FT = np.fft.rfft(W) #* 2 / n

print(np.allclose(np.array([[i.real, i.imag] for i in FT]), I))

print(f, p, ">>", np.round(P[f], 4))

for i in range(freqs):
  print(np.round(FT[i].real), ", ", np.round(FT[i].imag)," | ", np.round(I[i,0]),", ", np.round(I[i,1]))
