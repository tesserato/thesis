import random
import numpy as np

from Helper import save_wav

random.seed(0)
n = 44100

X = np.arange(n)
Y = np.zeros(n)

f = 100
p = np.pi
Y = np.cos(p + 2 * np.pi * f * X / n)


save_wav(8000 * Y)