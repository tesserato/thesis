import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import wave
import numpy as np
from numpy.fft import rfft, irfft

def read_wav(path): # returns signal & fps
  wav = wave.open(path , 'r')
  signal = np.frombuffer(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps

def fit(X, Y, n, k, W=None):
  A = np.polyfit(X, Y, k, w=W)
  print(f"A={A}")
  X = np.arange(n)
  return np.polyval(A, X)

# W, fps = read_wav("test_samples/tom.wav")
# W = W[:600]                # <|<|<|<|
# W = W - np.average(W)
# W = W / np.max(np.abs(W))
# n = W.shape[0]

n = 20
f = 4.25
p = np.pi/4
a = 3
T = np.arange(n)
W = a * np.cos(p + 2 * np.pi * f * T / n)


# plt.plot(W, "r.-")
# plt.show()
# exit()

FT = rfft(W) * 2 / n
fourier_f = np.argmax(np.abs(FT))
fourier_p = np.angle(FT[fourier_f]) % (2 * np.pi)
fourier_a = np.abs(FT[fourier_f])

print(f"W: n={n} | frequency={fourier_f} | phase={np.round(fourier_p, 2 )} | amplitude={np.round(fourier_a, 2 )}")


assert(np.count_nonzero(W) == n)


res_f = 200
res_p = 100

max_f = 10#n / 2
max_p = 2 * np.pi

F = np.linspace(0, max_f, res_f)
P = np.linspace(0, max_p, res_p)

FP = np.zeros((res_f, res_p))

# print(P.shape, F.shape)

total = np.sum(np.abs(W))

for t in range(n):
  for i in range(res_f):
    for j in range(res_p):
      FP[i,j] += W[t] * np.cos(P[j] + 2*np.pi*F[i]*t/n)

FP = FP * 2 / n
  # plt.pcolormesh(FP, cmap="bwr")
  # plt.show()
  # exit()

ind = np.unravel_index(np.argmax(FP, axis=None), FP.shape)
int_f = F[ind[0]]
int_p = P[ind[1]]
int_a = FP[ind]

print(f"I: frequency={round(int_f, 2)} | phase={round(int_p, 2)} | amplitude={round(int_a, 2)}")

Wf = fourier_a * np.cos(fourier_p + 2 * np.pi * T * fourier_f / n)
Wi = int_a * np.cos(int_p + 2 * np.pi * T * int_f / n)

plt.hlines(0, 0, n, "c")
plt.stem(T, W, '.1', markerfmt='k.', basefmt='k')
plt.plot(Wf, "r.-")
plt.plot(Wi, "b.-")
plt.xlabel('i')
plt.ylabel('Amplitude')

print("F error", np.average(np.abs(W - Wf)))
print("I error", np.average(np.abs(W - Wi)))

plt.show()
exit()


plt.pcolormesh(FP, cmap="bwr")
plt.show()


plt.plot(F, np.sum(FP, 1), "k.")

plt.plot(np.abs(FT), "r.")
plt.show()

