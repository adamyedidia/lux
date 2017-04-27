from __future__ import division
import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as p

MAX_X = 1
MAX_Y = 1

OCCLUDER_X = 0.5
OCCLUDER_Y = 0.5

EPS = 1e-9

SIDE_LENGTH = 1000
DELTA_X = MAX_X/SIDE_LENGTH

def edgeMatrixEntry(lightX, viewX):
#    print lightX, viewX, OCCLUDER_Y * (lightX - viewX) + viewX, OCCLUDER_X, \
#        OCCLUDER_Y * (lightX - viewX) + viewX + EPS > OCCLUDER_X
    if OCCLUDER_Y * (lightX - viewX) + viewX + EPS >= OCCLUDER_X:
#        print lightX, viewX
        return 0
    else:
        return (MAX_Y * DELTA_X)/(pi * sqrt(MAX_Y*MAX_Y + (lightX - viewX)**2))

def edgeMatrixEntryGlancing(lightX, viewX):
    if lightX > viewX + EPS:
        return 0
    else:
        return (MAX_Y * DELTA_X)/(pi * sqrt(MAX_Y*MAX_Y + (lightX + viewX)**2))

responseList = []

for lightXLarge in range(SIDE_LENGTH):
    lightX = lightXLarge / SIDE_LENGTH
    responseList.append([])
    for viewXLarge in range(SIDE_LENGTH):
        viewX = viewXLarge / SIDE_LENGTH
        responseList[-1].append(edgeMatrixEntryGlancing(lightX, viewX))


#    print responseList[-1]

responseArray = np.array(responseList)
p.matshow(responseArray)
p.show()

responseArrayInv = np.linalg.inv(np.array(responseArray))

p.plot(responseArrayInv[0][2:])
p.show()

p.matshow(responseArrayInv)
p.show()

print responseArrayInv
print np.linalg.det(responseArray)
