import random
import numpy as np

from Helper import save_wav

random.seed(0)
n = 1000

X = np.arange(n)
Y = np.zeros(n)

f = 2
Y = np.cos(2 * np.pi * f * X / n)


save_wav(5000 * Y, f"Python/local_f={f}.wav")