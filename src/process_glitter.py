import pickle
from math import pi
import numpy as np
from scenegenerator import getXUnitVectorFromSphereAngles

INTEREST_THRESHOLD = 2

def pront(x):
    print x

def normalize(vec):
    return vec/np.linalg.norm(vec)

def addToHistogram(dic, entry):
    if entry in dic:
        dic[entry] += 1
    else:
        dic[entry] = 1

def addToHistogramWithPolarization(dic, entry):
    if entry[0] in dic:
        if dic[entry][1] == entry[1]:
            dic[entry][0] += 1
        else:
            dic[entry][0] += 1
            dic[entry][1] = "unpolarized"

    else:
        dic[entry[0]] = (1, entry[1])

def pruneHistogram(dic):
    for i in dic:
        if dic[i] < INTEREST_THRESHOLD:
            del dic[i]


def computeFlakePhiFromFirstReturn(pixelVector, histogramItem, relevantParams):
    PHOTONS_PER_PULSE, GLITTER_AREA, ROTATION_PERIOD, DETECTOR_COOLDOWN = \
        relevantParams

    glitterDistance = histogramItem[0]

    numPhotonsReturning = histogramItem[1][0]
    numPhotonsExpected = PHOTONS_PER_PULSE * GLITTER_AREA * ROTATION_PERIOD /
        (DETECTOR_COOLDOWN * 2 * pi * glitterDistance * glitterDistance)

    if numPhotonsReturning > numPhotonsExpected:
        pront("Warning: more photons than expected~!")
        return 0.

    else:
        return acos(numPhotonsReturning / numPhotonsExpected)

def deducePointUnambiguous(pixelLocation, pixelVector, histogram, relevantParams):
    histogramItems = histogram.items()
    histogramItems.sort(key = lambda x: x[0])

    if len(histogramItems) != 2:
        pront("Point rejected: wrong histogram size")
        return

    if histogramItems[0][1] != "unp":
        pront("Point rejected: polarized first return")
        return

    if histogramItems[1][1] == "unp":
        pront("Point rejected: unpolarized second return")
        return

    putativeFlakeLocation = pixelLocation + pixelLocation
    flakeTheta = histogramItems[1][1][1]
    putativeFlakePhi = computeFlakePhiFromFirstReturn(pixelVector, \
        histogramItems[0], relevantParams)

    likelyFlakeNormal = 


# (time, polarization)
def constructHistogramArray(listOfData, experimentParams):
    PIXELS_TO_A_SIDE = experimentParams["PIXELS_TO_A_SIDE"]
    CAMERA_LOCATION = experimentParams["CAMERA_LOCATION"]
    CAMERA_SIDE_LENGTH = experimentParams["CAMERA_SIDE_LENGTH"]
    PIXEL_SIDE_LENGTH = experimentParams["PIXEL_SIDE_LENGTH"]
    PULSE_WIDTH = experimentParams["PULSE_WIDTH"]
    PULSE_HEIGHT = experimentParams["PULSE_HEIGHT"]
    PHOTONS_PER_PULSE = experimentParams["PHOTONS_PER_PULSE"]
    GLITTER_AREA = experimentParams["GLITTER_AREA"]
    ROTATION_PERIOD = experimentParams["ROTATION_PERIOD"]
    DETECTOR_COOLDOWN = experimentParams["DETECTOR_COOLDOWN"]

    cameraNegativeCornerLocation = CAMERA_LOCATION + \
        np.array([0, -CAMERA_SIDE_LENGTH/2., -CAMERA_SIDE_LENGTH/2.])

    pulseNegativeCornerLocation = np.array([0, -PULSE_WIDTH/2., -PULSE_HEIGHT/2.])

    histogramArray = []

    for i in PIXELS_TO_A_SIDE:
        histogramArray.append([])
        for j in PIXELS_TO_A_SIDE:
            histogramArray[-1].append({})

    for snapshot in listOfData:
        for i in PIXELS_TO_A_SIDE:
            for j in PIXELS_TO_A_SIDE:
                addToHistogram(histogramArray[i][j], int(snapshot[i][j][0]))

    for i in PIXELS_TO_A_SIDE:
        for j in PIXELS_TO_A_SIDE:
            pruneHistogram(histogramArray[i][j])

    listOfUnambiguousPixelInfo = []
    # Ignore all ambiguous pixels!

    for i in PIXELS_TO_A_SIDE:
        for j in PIXELS_TO_A_SIDE:
            if len(histogramArray[i][j]) == 2:

                listOfUnambiguousPixelInfo.append(((i, j), histogramArray[i][j]))

    for unambiguousPixelInfo in listOfUnambiguousPixelInfo:
        pixelCoords = unambiguousPixelInfo[0]
        histogram = unambiguousPixelInfo[1]

        pixelI = pixelCoords[0]
        pixelJ = pixelCoords[1]

        pixelLocation = cameraNegativeCornerLocation + \
            np.array([0, (pixelI+0.5)*PIXEL_SIDE_LENGTH, \
            (pixelJ+0.5)*PIXEL_SIDE_LENGTH])

        pulseCenterLocation = pulseNegativeCornerLocation + \
            np.array([0, (pixelI+0.5)*PULSE_WIDTH/PIXELS_TO_A_SIDE, \
            (pixelJ+0.5)*PULSE_WIDTH/PIXELS_TO_A_SIDE])

        pixelVector = pulseCenterLocation - pixelLocation

        deducePointUnambiguous(pixelLocation, pixelVector, histogram,
            (PHOTONS_PER_PULSE, GLITTER_AREA, ROTATION_PERIOD, DETECTOR_COOLDOWN))
