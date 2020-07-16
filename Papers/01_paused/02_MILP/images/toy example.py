import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import wave
import numpy as np
from numpy.fft import rfft, irfft
import random

random.seed(0)
n = 100

T = np.arange(n)
W = np.zeros(n)


for _ in range(100):
  a = random.uniform(1, 5)
  f = random.uniform(1, 10)
  p = random.uniform(0, 2 * np.pi)
  W += a * np.cos(p + 2 * np.pi * f * T / n)

res_f = 50
res_p = 10

max_f = n / 2
max_p = 2 * np.pi

F = np.linspace(0, max_f, res_f)
P = np.linspace(0, max_p, res_p)

FP = np.zeros((res_f, res_p))

# print(P.shape, F.shape)

# total = np.sum(np.abs(W))

for t in range(n):
  for i in range(res_f):
    for j in range(res_p):
      FP[i,j] += W[t] * np.cos(P[j] + 2*np.pi*F[i]*t/n)

# for t in range(1, n+1):
#   for k in range(1, t + 1):
#     f0 = n * k / t #- n * p / (2 * np.pi * t)
#     f2pi = n * k / t - n * 2 * np.pi / (2 * np.pi * t)
#     plt.plot([0, 2 * np.pi], [f0, f2pi], "k-", alpha=1-(1/n*t))

FP = FP * 2 / n

ind = np.unravel_index(np.argmax(np.abs(FP), axis=None), FP.shape)
int_a = FP[ind]
if int_a > 0:
  int_f = F[ind[0]]
  int_p = P[ind[1]]
else:
  print("!!")
  int_a *= -1
  int_f = F[ind[0]]
  int_p = P[ind[1]] + np.pi

print(f"IN: frequency={round(int_f, 2)} | phase={round(int_p, 2)} | amplitude={round(int_a, 2)}")

FT = rfft(W) * 2 / n
fourier_f = np.argmax(np.abs(FT))
fourier_p = np.angle(FT[fourier_f]) #% (2 * np.pi)
fourier_a = np.abs(FT[fourier_f])
print(f"FT: frequency={fourier_f} | phase={np.round(fourier_p, 2 )} | amplitude={np.round(fourier_a, 2 )}")


Wf = fourier_a * np.cos(fourier_p + 2 * np.pi * T * fourier_f / n)
Wi = int_a * np.cos(int_p + 2 * np.pi * T * int_f / n)

print("F error", np.average(np.abs(W - Wf)))
print("I error", np.average(np.abs(W - Wi)))

plt.plot(np.abs(FT), "r.-")
plt.plot(F, np.average(np.abs(FP), 1), "b.-")
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

plt.hlines(0, 0, n, "c")
plt.stem(T, W, '.1', markerfmt='k.', basefmt='k')
plt.plot(Wf, "r-")
plt.plot(Wi, "b-")
plt.xlabel('i')
plt.ylabel('Amplitude')
plt.show()

plt.tight_layout()
plt.title( f"n={n}")
plt.xlabel('Phase [Radians]')
plt.ylabel('Frequency [Hz]')
plt.pcolormesh(P, F, FP, cmap="bwr")#, antialiased=True, shading='gouraud')
plt.plot(int_p, int_f, "ko")
plt.show()


