from math import sqrt, pi, sin, cos, cosh, sinh, acosh, asinh
import numpy as np
import matplotlib.pyplot as p
import random
from jack3d import 

def triangleResponse(xd, v0, v1, v2, detectorNormal):
    def approxFunc(t):
        source = np.array([0,0,0])
        detector = np.array([xd,0,0])
