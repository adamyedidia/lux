from __future__ import division
import numpy as np
from PIL import Image
from video_magnifier import viewFrame, viewFrameR
from math import ceil, floor
from cr2_processor import convertRawFileToArray
import pickle

oldRectify = False
rawRectify = False
wrongRectify = False
veryWrongRectify = False
seeObs = True


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

def makeGrid(corner, p1, p2, oppCorner, steps1, steps2):

    returnArray = []

    for i in range(steps1):
        returnArray.append([])

        print "Making grid:", i, "/", steps1

        vec2 = (i/steps1 * (oppCorner - p2) + (steps1 - i)/steps1 * (p1 - corner))/steps2



        for j in range(steps2):

#            vec1 = (j/steps2 * (oppCorner - p1) + (steps2 - j)/steps2 * (p2 - corner))/steps1
            vec1 = (p2 - corner)/steps1

#            print i, j, vec1, vec2, corner + i*vec1 + j*vec2

            returnArray[-1].append(corner + i*vec1 + j*vec2)

    return returnArray

def rectify(arr, corner, p1, p2, oppCorner, steps1, steps2):

    grid = makeGrid(corner, p1, p2, oppCorner, steps1, steps2)

    returnArray = []

    for j in range(steps2):
        returnArray.append([])

        print "Rectifying:", j, "/", steps2

        for i in range(steps1):
            vec = grid[i][j]

            value = fuzzyLookup(arr, (vec[1], vec[0]))

            returnArray[-1].append(value)

    return np.array(returnArray)

if oldRectify:

    rectifiedFrame = rectify(frame, CORNER, P1, P2, OPP_CORNER, 1500, 1000)

    viewFrame(np.flip(rectifiedFrame, 1))

if seeObs:
    dirName = "calibration"

    path = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified.p"

    print "Loading..."
    arr = pickle.load(open(path, "r"))
    print "Loaded!"

    viewFrame(arr)

if rawRectify:

#    FILE_NAME = "/Users/adamyedidia/flags_garbled/texas_garbled/IMG_5048.CR2"
#    arr = convertRawFileToArray(FILE_NAME)

    dirName = "uk_garbled_small"

    path = "/Users/adamyedidia/flags_garbled/" + dirName + "/average.p"

    print "Loading..."
    arr = pickle.load(open(path, "r"))
    print "Loaded!"

    CORNER = np.array([1456, 2830])
    P2 = np.array([5130, 2916])
    OPP_CORNER = np.array([5484, 822])
    P1 = np.array([1185, 689])

#    croppedArr = arr[0:100, 0:100]

#    CORNER = np.array([0, 0])
#    P1 = np.array([100, 0])
#    OPP_CORNER = np.array([0, 100])
#    P2 = np.array([100, 100])

    rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 5000, 3000)

    path2 = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified.p"


#    viewFrame(rectifiedArr)

    print "Writing..."
    pickle.dump(rectifiedArr, open(path2, "w"))
    print "Done!"

#    viewFrame(arr)

if wrongRectify:

    dirName = "texas_garbled"

    path = "/Users/adamyedidia/flags_garbled/" + dirName + "/average.p"

    print "Loading..."
    arr = pickle.load(open(path, "r"))
    print "Loaded!"

    CORNER = np.array([1300, 2900])
    P2 = np.array([5300, 2900])
    OPP_CORNER = np.array([5300, 700])
    P1 = np.array([1300, 700])

    rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 5000, 3000)

    path2 = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified_wrong.p"

    print "Writing..."
    pickle.dump(rectifiedArr, open(path2, "w"))
    print "Done!"

if veryWrongRectify:

    dirName = "texas_garbled"

    path = "/Users/adamyedidia/flags_garbled/" + dirName + "/average.p"

    print "Loading..."
    arr = pickle.load(open(path, "r"))
    print "Loaded!"

#    CORNER = np.array([1300, 2900])
#    P2 = np.array([5300, 2900])
#    OPP_CORNER = np.array([5300, 700])
#    P1 = np.array([1300, 700])

    CORNER = np.array([3300, 2900])
    P2 = np.array([5300, 1800])
    OPP_CORNER = np.array([3300, 700])
    P1 = np.array([1300, 1800])

    rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 5000, 3000)

    path2 = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified_very_wrong.p"

    print "Writing..."
    pickle.dump(rectifiedArr, open(path2, "w"))
    print "Done!"
