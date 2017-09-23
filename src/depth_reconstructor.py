from __future__ import division
import numpy as np
from PIL import Image
from video_magnifier import viewFrame, viewFrameR
from video_processor import padIntegerWithZeros
from math import ceil, floor
from cr2_processor import convertRawFileToArray
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import pickle
import scipy.io
import sys
import matplotlib.pyplot as p

doStuff = True
makeBoundariesTest = False

REAL_PERSON_SPEED = 7
REAL_PERSON_START_PROGRESS = 1100
PIXELS_PER_INCH_HORIZ = 2
PIXELS_PER_INCH_VERT = 15
DOOR_WIDTH = 43 # inches
HALLWAY_WIDTH = 89 # inches

MAX_IMG_PIXEL = 1353

LOCATION_0 = np.array([391,531])
LOCATION_1 = np.array([1801, 478])
LOCATION_2 = np.array([844, 1186])

LOCATIONS = [LOCATION_0, LOCATION_1, LOCATION_2]

def moreThanZero(x):
    return x >= -10



def findRealPersonLocation(progress, currentLeg=0):
    nextLeg = (currentLeg + 1)%3

    start = LOCATIONS[currentLeg]
    end = LOCATIONS[nextLeg]
    dist = np.linalg.norm(end-start)

    if dist < progress:
        return findRealPersonLocation(progress - dist, nextLeg)

    else:
        return start + (end - start)*progress/dist


def findLongestRun(arr, funcSatisfied):
    inRun = False
    runStart = None
    runEnd = None
    longestRunStart = None
    longestRunEnd = None
    longestRunLength = 0

    for i, val in enumerate(arr):
#        print funcSatisfied(val)
        if funcSatisfied(val):
            if not inRun:
#                print "Started run!"
                runStart = i
                inRun = True

        else:
            if inRun:
                runEnd = i - 1
                runLength = runEnd - runStart + 1
#                print "Ended", runLength, "length run."
                if runLength > longestRunLength:
                    longestRunStart = runStart
                    longestRunEnd = runEnd
                    longestRunLength = runLength

                inRun = False

    i = len(arr)

    if inRun:
        runEnd = i - 1
        runLength = runEnd - runStart + 1

#        print runStart, runEnd, runLength

        if runLength > longestRunLength:
            longestRunStart = runStart
            longestRunEnd = runEnd
            longestRunLength = runLength

    return longestRunLength, longestRunStart, longestRunEnd

def getLocation(longestRunLength, longestRunStart, longestRunEnd):
    depth = DOOR_WIDTH * HALLWAY_WIDTH / (1 + longestRunLength / PIXELS_PER_INCH_HORIZ)
    offset = (longestRunStart + longestRunEnd)/(2*PIXELS_PER_INCH_VERT)
    return depth, offset

def convertToLocation(longestRunLength, longestRunStart, longestRunEnd):
#    print longestRunLength, longestRunStart, longestRunEnd
    if longestRunLength < 100:
        return None

    elif longestRunStart == 0 or longestRunEnd == MAX_IMG_PIXEL:
        if longestRunLength < 400:
            return None
        else:
            return getLocation(longestRunLength, longestRunStart, longestRunEnd)

    else:
        return getLocation(longestRunLength, longestRunStart, longestRunEnd)

def xFunc(loc):
    return 10*(210-2.1*loc[1])

def yFunc(loc):
    return 25*(loc[0] + 15)

def computeBoundaryDot(runLength, runStart, runEnd):
    assert runStart == 1 or runEnd == MAX_IMG_PIXEL-1

    return convertToLocation(runLength, runStart, runEnd)

def computeFirstBoundaryLine(im):
    runStart = 1

    xs = []
    ys = []

    for runEnd in range(2, MAX_IMG_PIXEL):
        runLength = runEnd - runStart + 1
        dot = computeBoundaryDot(runLength, runStart, runEnd)

        if not dot == None:

            x, y = xFunc(dot), yFunc2(dot)

            if x > 100 and y > 100 and x < im.shape[1]-100 and y < im.shape[0]-100:

                xs.append(x)
                ys.append(y)

    p.plot(xs, ys, c='c', linestyle=':')

def computeSecondBoundaryLine(im):
    runEnd = MAX_IMG_PIXEL-1

    xs = []
    ys = []

    for runStart in range(1, MAX_IMG_PIXEL-1):
        runLength = runEnd - runStart + 1
        dot = computeBoundaryDot(runLength, runStart, runEnd)

        if not dot == None:

            x, y = xFunc(dot), yFunc2(dot)

            if x > 100 and y > 100 and x < im.shape[1]-100 and y < im.shape[0]-100:

                xs.append(x)
                ys.append(y)

    p.plot(xs, ys, c='c', linestyle=':')

if __name__ == "__main__":
    if makeBoundariesTest:
        im = p.imread("/Users/adamyedidia/Desktop/hallway_image.png")
        implot = p.imshow(im)

        computeFirstBoundaryLine(im)
        computeSecondBoundaryLine(im)

        p.show()

    if doStuff:

        imRaw = Image.open("/Users/adamyedidia/Desktop/hallway_walking_movie.png")

        im = np.array(imRaw).astype(float)

        imR = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)[0]
        imB = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)[2]

    #    viewFrameR(imR-1.5*imB, magnification=2, differenceImage=True)

        adjustedIm = imR-1.5*imB

        newArr = []

        for oneDArr in adjustedIm:
            newArr.append(gaussian_filter(oneDArr, sigma=40))

    #    viewFrameR(np.array(newArr), magnification=2, differenceImage=True)

        im = p.imread("/Users/adamyedidia/Desktop/hallway_image.png")

        frameCounter = 0

        progress = REAL_PERSON_START_PROGRESS


        for i, oneDArr in enumerate(newArr):

            progress += REAL_PERSON_SPEED

            if i == 400:
                REAL_PERSON_SPEED = 4

            if i == 700:
                REAL_PERSON_SPEED = 7

            if i % 2 == 0:
                print i

                #print loc

                p.clf()
                implot = p.imshow(im)

                computeFirstBoundaryLine(im)
                computeSecondBoundaryLine(im)

                realPersonLocation = findRealPersonLocation(progress)

                p.scatter(realPersonLocation[0], realPersonLocation[1], c="b")

                longestRunLength, longestRunStart, longestRunEnd = \
                    findLongestRun(oneDArr, moreThanZero)

                loc = convertToLocation(longestRunLength, longestRunStart, longestRunEnd)

                if not loc == None:
                    x = xFunc(loc)
                    y = yFunc(loc)

                if not loc == None and x > 100 and y > 100 and x < im.shape[1]-100 and \
                    y < im.shape[0]-100:
    #                print type(137), type(int(200-loc[1]))
    #                print int(200-loc[1]), int(89-loc[0])

                    p.scatter([x], [y], c='r')

                p.savefig("hallway_movie/hallmov" + padIntegerWithZeros(frameCounter, 3) + ".png")
                frameCounter += 1
    #            print longestRunLength, \
    #                convertToLocation(longestRunLength, longestRunStart, longestRunEnd)

    #            print oneDArr

    #            viewFrameR(np.array([oneDArr]*100), magnification=2, differenceImage=True)
