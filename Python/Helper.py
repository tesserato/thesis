import numpy as np
import random
import wave
from numpy.fft import rfft, irfft

from scipy.spatial import Delaunay


def read_wav(path): 
  """returns signal & fps"""
  wav = wave.open(path , 'r')
  signal = np.frombuffer(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps

def save_wav(signal, name = 'test.wav', fps = 44100): #save .wav file to program folder
  o = wave.open(name, 'wb')
  o.setframerate(fps)
  o.setnchannels(1)
  o.setsampwidth(2)
  o.writeframes(np.int16(signal)) # Int16
  o.close()






#############################################################
#############################################################
#############################################################

def get_delaunay(X, Y):
  points = np.array([[x, y] for x, y in zip(pos_X, pos_Y)])
  tri = Delaunay(points)
  X_tri = []
  Y_tri = []
  for ia, ib, ic in tri.vertices:
    xa, ya = points[ia]
    xb, yb = points[ib]
    xc, yc = points[ic]
    X_tri.append(xa), Y_tri.append(ya)
    X_tri.append(xb), Y_tri.append(yb)
    X_tri.append(xc), Y_tri.append(yc)
    X_tri.append(xa), Y_tri.append(ya)
    X_tri.append(None), Y_tri.append(None)
  return X_tri, Y_tri


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n, 2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def get_curvature_3_points(X, Y):
  '''Return max curvatures, avg and r for the poly approximation of 3 pulses amplitude at a time'''
  curvatures = []
  for i in range(len(X) - 2):
    x0, x1, x2 = X[i], X[i + 1], X[i + 2]
    y0, y1, y2 = Y[i], Y[i + 1], Y[i + 2]
    x, y = np.array([x0, x1, x2]), np.array([y0, y1, y2])
    # y = y / y1
    a0, a1, a2 = poly.polyfit(x, y, 2)
    # c = np.abs(2 * a2) # max curvature
    c = ((2 * a2 * x1 + a1) / np.sqrt((2 * a2 * x1 + a1)**2 + 1) - (2 * a2 * x0 + a1) / np.sqrt((2 * a2 * x0 + a1)**2 + 1)) / (x1-x0)
    # c = 2 * a2 / ((a1 + 2*a2*x1)**2 + 1)**(3/2)
    curvatures.append(c)
    # X = np.linspace(x0, x2, 100)
    # Y = poly.polyval(X, args)
    # for x, y in zip(X, Y):
    #   pos_curvature_X.append(x); pos_curvature_Y.append(y)
    # pos_curvature_X.append(None) ; pos_curvature_Y.append(None)
  average_curvature = np.abs(np.average(curvatures))
  radius = 1 / average_curvature
  curvatures = [curvatures[0]] + curvatures + [curvatures[-1]]
  return curvatures, average_curvature, radius

def smoothstep(t):

  # return 3 * t**2 - 2 * t**3
  # return 6 * t**5 - 15 * t**4 + 10 * t**3
  # return -20 * t**7 + 70 * t**6 - 84 * t**5 + 35 * t**4
  return t

def old_smooth(X, Y, n):

  Yn = np.zeros(len(Y))

  Yn[0] = (2 * np.sqrt(Y[0]) + np.sqrt(Y[1]))**2 / 9
  Yn[-1] = (2 * np.sqrt(Y[-1]) + np.sqrt(Y[-2]))**2 / 9
  
  for i in range(1, len(Y) - 1):
    Yn[i] = (np.sqrt(Y[i - 1]) + np.sqrt(Y[i]) + np.sqrt(Y[i +1]))**2 / 9
  Y[:] = Yn[:]
  
  X[0] = 0
  Xs = np.arange(n)
  Ye0 = np.zeros(n)
  Ye1 = np.zeros(n)
  Yo0 = np.zeros(n)
  Yo1 = np.zeros(n)

  t = np.linspace(0, .5, X[1])[::-1]
  w = smoothstep(t)
  Yo1[0 : X[1]] = w * Y[0]
  Yo0[0 : X[1]] = (1 - w) * Y[1]

  for i in range(0, len(X) - 5, 4):
    x0, x1, x2, x3, x4, x5 = X[i], X[i + 1], X[i + 2], X[i + 3], X[i + 4], X[i + 5]
    y0, y1, y2, y3, y4, y5 = Y[i], Y[i + 1], Y[i + 2], Y[i + 3], Y[i + 4], Y[i + 5]

    '''even'''
    t = np.linspace(0, 1, x2 - x0)
    w = smoothstep(t)
    Ye0[x0 : x2] = y0 * (1 - w)
    Ye1[x0 : x2] = y2 * w
    
    t = np.linspace(0, 1, x4 - x2)[::-1]
    w = smoothstep(t)
    Ye0[x2 : x4] = y4 * (1 - w)
    Ye1[x2 : x4] = y2 * w
    
    '''odd'''
    t = np.linspace(0, 1, x3 - x1)
    w = smoothstep(t)
    Yo0[x1 : x3] = y1 * (1 - w)
    Yo1[x1 : x3] = y3 * w
    
    t = np.linspace(0, 1, x5 - x3)[::-1]
    w = smoothstep(t)
    Yo0[x3 : x5] = y5 * (1 - w)
    Yo1[x3 : x5] = y3 * w
    

  t = np.linspace(0, .5, n - X[-4])
  w = smoothstep(t)
  Yo1[X[-4] : ] = w * Y[-1]
  Yo0[X[-4] : ] = (1 - w) * Y[-4]

  t = np.linspace(0, 1, n - X[-5])
  w = smoothstep(t)
  Ye1[X[-5] : ] = w * Y[-2]
  Ye0[X[-5] : ] = (1 - w) * Y[-5]

  YY = (Ye0 + Ye1 + Yo0 + Yo1) / 2
  return Xs, YY, Ye0, Ye1, Yo0, Yo1
