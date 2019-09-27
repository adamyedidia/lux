
import matplotlib.pyplot as p
import numpy as np
from math import floor, ceil
import random

SCENE_LENGTH = 64

NOISE = 1e-10

LIGHT_TO_OCCLUDER_DISTANCE = 0.5

def emptyOccluder():
    return np.array([0 for _ in range(SCENE_LENGTH)])

def edgeOccluder(boundary):
    return np.array([1*(i<boundary) for i in range(SCENE_LENGTH)])
    
def centralObjectOccluder(radius):
    return np.array([1*(abs(i-SCENE_LENGTH/2)<=radius) for i in range(SCENE_LENGTH)])

def randomOccluder():
    return np.array([1*(random.random() < 0.5) for _ in range(SCENE_LENGTH)])

def periodicOccluder(period):
    return np.array([1*(i%period<period/2) for i in range(SCENE_LENGTH)])

# index is a float
def fuzzyLookup(array, index):
    floorIndex = int(floor(index))
    ceilIndex = int(ceil(index))
    
    residue = index % 1
    
    arrayBelow = array[floorIndex]
    arrayAbove = array[ceilIndex]
    
    return (1-residue) * arrayBelow + residue * arrayAbove

# As a function of the light's index, get the pattern of light on the detector array
def occlude(occluderArray, lightIndex):
    detectorArray = []
    
    for i in range(SCENE_LENGTH):
        detectorArray.append(fuzzyLookup(occluderArray, (lightIndex + i)/2))

    return detectorArray
    
def buildTheOccluderMatrix(occluderArray):
    occluderMatrix = []
    
    for i in range(SCENE_LENGTH):
        occluderMatrix.append(occlude(occluderArray, i))
        
    return np.array(occluderMatrix)
    
def getSVDForOccluderMatrix(occluderMatrix):
    return np.linalg.svd(occluderMatrix, full_matrices=False, \
        compute_uv=False)

def evaluateOccluder(occluderArray):
    occluderMatrix = buildTheOccluderMatrix(occluderArray)
    print(occluderMatrix)
    print(np.linalg.det(occluderMatrix))
    
 #   svd = getSVDForOccluderMatrix(occluderMatrix)
 #   print svd
    
def plotSVDOfOccluder(occluderArray):
    occluderMatrix = buildTheOccluderMatrix(occluderArray)

    svd = getSVDForOccluderMatrix(occluderMatrix)

    p.semilogy(svd)
#    p.show()
    

period = 16

#print periodicOccluder(period)

#print np.fft.fft(periodicOccluder(period))

evaluateOccluder(periodicOccluder(period))
evaluateOccluder(randomOccluder())
evaluateOccluder(edgeOccluder(SCENE_LENGTH/2))

plotSVDOfOccluder(periodicOccluder(period))
plotSVDOfOccluder(randomOccluder())    
plotSVDOfOccluder(edgeOccluder(SCENE_LENGTH/2))
ax = p.gca()
ax.set_ylim([1e-5, 1e3])

#plotSVDOfOccluder(centralObjectOccluder(SCENE_LENGTH/4))
p.show()

#p.plot(emptyOccluder())
#p.plot(edgeOccluder(SCENE_LENGTH/2))
#p.plot(centralObjectOccluder(SCENE_LENGTH/4))
#p.show()