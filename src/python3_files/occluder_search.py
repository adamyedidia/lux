import matplotlib.pyplot as p
import numpy as np
from math import floor, ceil
import random


SCENE_HEIGHT = 20
SCENE_WIDTH = 20

NOISE = 1e-10

LIGHT_TO_OCCLUDER_DISTANCE = 0.5

def createEmptyMatrix():
    mat = np.array([[0]*SCENE_WIDTH for _ in range(SCENE_HEIGHT)])

    return mat

# radius from (0,0)
def circularOccluder(radius):
    mat = np.array([[1*(i*i+j*j<radius*radius) for i in range(SCENE_WIDTH)]
        for j in range(SCENE_HEIGHT)])

    return mat

def squareOccluder():
    mat = np.array([[1*((i<=SCENE_HEIGHT/2.) and (j<=SCENE_WIDTH/2.)) for j in \
        range(SCENE_WIDTH)] for i in range(SCENE_HEIGHT)])

    return mat

def dotOccluder():
    mat = np.array([[1*((i==SCENE_HEIGHT/2 or i==SCENE_HEIGHT/2-1) and \
                        (j==SCENE_WIDTH/2 or j==SCENE_WIDTH/2-1)) for j in \
                        range(SCENE_WIDTH)] for i in range(SCENE_HEIGHT)])

    return mat

def checkerboardOccluder():
    mat = np.array([[((i+j)%2)==0 for j in range(SCENE_WIDTH)] for i in range(SCENE_HEIGHT)])

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

def flatten(mat):
    outList = []

    for row in mat:
        for item in row:
            outList.append(item)

    return outList

# for each possible light source location, make a vector for the response
def buildTheOccluderMetaMatrix(occluderMatrix):
    listOfRows = []

    for lightI in range(SCENE_HEIGHT):
        for lightJ in range(SCENE_WIDTH):
            detectorMatrix = occlude(occluderMatrix, (lightI, lightJ))

#            print flatten(detectorMatrix)

            listOfRows.append(flatten(detectorMatrix))

#    print listOfRows

    return np.array(listOfRows)

def getSVDForMetaMatrix(occluderMetaMatrix):
    return np.linalg.svd(occluderMetaMatrix, full_matrices=False, \
        compute_uv=False)

# processEigenvalues is a function that you pass in as an argument
def evaluateOccluder(occluderMatrix, processEigenvalues):
    mm = buildTheOccluderMetaMatrix(occluderMatrix)

    svd = getSVDForMetaMatrix(mm)

    return processEigenvalues(svd)

def countEValuesAboveNoise(eigs):
    return sum([1*(i>NOISE) for i in eigs])

def deepCopy(mat):
    return [i[:] for i in mat]

def flipBit(mat, coords):
    i, j = coords

    mat[i][j] = 1-mat[i][j]

def plotSVDOfOccluder(occluderMatrix):
    mm = buildTheOccluderMetaMatrix(occluderMatrix)

    svd = getSVDForMetaMatrix(mm)

    p.semilogy(svd)
    p.show()


# greedy random search
def occluderSearch(guess, processEigenvalues, maxStagnationCount):
    occluderMatrix = deepCopy(guess)
    currentEvaluation = evaluateOccluder(occluderMatrix, processEigenvalues)

    stagnationCount = 0

    while stagnationCount < maxStagnationCount:

        randomCoords = (random.randint(0, SCENE_HEIGHT-1), random.randint(0, SCENE_WIDTH-1))
        flipBit(occluderMatrix, randomCoords)

        # See if this version is better; if not, switch back
        newEvaluation = evaluateOccluder(occluderMatrix, processEigenvalues)

        if newEvaluation > currentEvaluation:
            # Keep the new one
            currentEvaluation = newEvaluation
            stagnationCount = 0

        else:
            # Switch back to the old
            flipBit(occluderMatrix, randomCoords)
            stagnationCount += 1

        p.matshow(occluderMatrix)
        p.show()

        print(currentEvaluation)



    return occluderMatrix

#occluderMatrix = createEmptyMatrix()

#bestOccluder = occluderSearch(createEmptyMatrix(), countEValuesAboveNoise, 5)

#p.matshow(bestOccluder)
#p.savefig("best_occluder.png")

#detectorMatrix = occlude(occluderMatrix, (90, 10))

#p.matshow(detectorMatrix, cmap=p.cm.gray)
#p.show()

checkerMat = checkerboardOccluder()
p.matshow(checkerMat)
p.savefig("checker_occluder.png")
#p.show()

p.clf()
plotSVDOfOccluder(checkerMat)

dotMat = dotOccluder()
p.matshow(dotMat)
p.savefig("dot_occluder.png")
#p.matshow(occluderMatrix)
#p.show()
plotSVDOfOccluder(dotMat)

squareMat = squareOccluder()
p.matshow(squareMat)
p.savefig("square_occluder.png")

plotSVDOfOccluder(squareMat)

p.clf()
circleMat = circularOccluder(12)
p.matshow(circleMat)
p.savefig("circle_occluder.png")
plotSVDOfOccluder(circleMat)

#plotSVDOfOccluder(bestOccluder)

#p.show()

#ax = p.gca()
#ax.set_ylim(ymin=1e-10)
#p.show()

#print evaluateOccluder(occluderMatrix, countEValuesAboveNoise)
