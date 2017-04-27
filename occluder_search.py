import matplotlib.pyplot as p
import numpy as np
from math import floor, ceil

SCENE_HEIGHT = 100
SCENE_WIDTH = 100

LIGHT_TO_OCCLUDER_DISTANCE = 0.5

def createEmptyMatrix():
    mat = np.array([[0.]*SCENE_WIDTH for _ in range(SCENE_HEIGHT)])

    return mat

# radius from (0,0)
def circularOccluder(radius):
    mat = np.array([[1*(i*i+j*j<radius*radius) for i in range(SCENE_WIDTH)]
        for j in range(SCENE_HEIGHT)])

    return mat

def squareOccluder():
    mat = np.array([[1*((i<=SCENE_WIDTH/2.) and (j<=SCENE_HEIGHT/2.)) for i in \
        range(SCENE_WIDTH)] for j in range(SCENE_HEIGHT)])

    return mat

# index into the matrix, but with a float, weighting by how close you
# are to each adjacent number
def fuzzyLookup(occluderMatrix, indices):
    i, j = indices
    floorI = int(floor(i))
    ceilI = int(ceil(i))
    floorJ = int(floor(j))
    ceilJ = int(ceil(j))

    residueI = i % 1
    residueJ = j % 1

    floorIfloorJ = occluderMatrix[floorI][floorJ]
    floorIceilJ = occluderMatrix[floorI][ceilJ]
    ceilIfloorJ = occluderMatrix[ceilI][floorJ]
    ceilIceilJ = occluderMatrix[ceilI][ceilJ]

    return (1-residueI) * (1-residueJ) * floorIfloorJ + \
        (1-residueI) * residueJ * floorIceilJ + \
        residueI * (1-residueJ) * ceilIfloorJ + \
        residueI * residueJ * ceilIceilJ

# As a function of where the light source is and the occluder pattern,
# get the pattern of light on the detector array
def occlude(occluderMatrix, lightCoords):
    lightI, lightJ = lightCoords

    detectorMatrix = []

    for sceneI in range(SCENE_HEIGHT):
        detectorMatrix.append([])
        for sceneJ in range(SCENE_WIDTH):
            occluderI = LIGHT_TO_OCCLUDER_DISTANCE * sceneI + \
                (1-LIGHT_TO_OCCLUDER_DISTANCE) * lightI

            occluderJ = LIGHT_TO_OCCLUDER_DISTANCE * sceneJ + \
                (1-LIGHT_TO_OCCLUDER_DISTANCE) * lightJ

            detectorMatrix[-1].append(1-fuzzyLookup(occluderMatrix, \
                (occluderI, occluderJ)))

    return detectorMatrix

occluderMatrix = circularOccluder(65)

detectorMatrix = occlude(occluderMatrix, (90, 10))

p.matshow(detectorMatrix, cmap=p.cm.gray)
p.show()

occluderMatrix = createEmptyMatrix()
