from __future__ import division
import numpy as np
from PIL import Image
from video_magnifier import viewFrame, viewFrameR
from math import ceil, floor
from cr2_processor import convertRawFileToArray
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import pickle
import scipy.io
import sys

oldRectify = False
rawRectify = True
wrongRectify = False
veryWrongRectify = False
seeObs = False
subtractRectify = False
subtractRectify2 = False
hallwayImaging = False

def displayFlattenedFrame(flattenedFrame, height, magnification=1, \
    differenceImage=False, filename=None):

    frame = np.array([flattenedFrame]*height)
    viewFrame(frame, differenceImage=differenceImage, magnification=magnification,
        filename=filename)

def blur2DImage(arr, blurRadius):
#    print arr.shape

    rearrangedIm = np.swapaxes(np.swapaxes(arr, 0, 2), 1, 2)
    imageRed = rearrangedIm[0]
    imageGreen = rearrangedIm[1]
    imageBlue = rearrangedIm[2]

    imageRed = gaussian_filter(imageRed, blurRadius, truncate=4.)
    imageGreen = gaussian_filter(imageGreen, blurRadius, truncate=4.)
    imageBlue = gaussian_filter(imageBlue, blurRadius, truncate=4.)

    blurredImage = np.swapaxes(np.swapaxes(np.array([imageRed, imageGreen, \
        imageBlue]), 1, 2), 0, 2)

    return blurredImage

def average(x):
    return sum(x)/len(x)

def flattenFrame(frame):
    rearrangedFrame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)
    outOfOrderFlattenedFrame = np.array([average(i) for i in rearrangedFrame])
    return np.swapaxes(outOfOrderFlattenedFrame, 0, 1)

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

if __name__ == "__main__":

    if subtractRectify:
        dirName = "b_dark"
        path = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/" + dirName + \
            "/average.p"

        print "Loading..."
        arr = pickle.load(open(path, "r"))
        print "Loaded!"

        CORNER = np.array([120, 309])
        P1 = np.array([119, 302])
        P2 = np.array([415, 193])
        OPP_CORNER = np.array([416, 200])

        print arr.shape

        rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 500, 300)

    #    viewFrame(rectifiedArr)

        dirName2 = "back_dark"
        path2 = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/" + dirName2 + \
            "/average.p"

        print "Loading..."
        arr = pickle.load(open(path2, "r"))
        print "Loaded!"

        subtractArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 500, 300)

    #    viewFrame(subtractArr)

        adjustedArr = rectifiedArr - subtractArr

        viewFrame(adjustedArr, 1e1, differenceImage=True)

        displayFlattenedFrame(flattenFrame(adjustedArr), 100, 1e1, differenceImage=True)

        path3 = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/" + dirName + \
            "/rectified.p"

        matPath = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/" + dirName + \
            "/arr.mat"

        print "Writing..."
        pickle.dump(flattenFrame(adjustedArr), open(path3, "w"))
        scipy.io.savemat(matPath, dict(arr=flattenFrame(adjustedArr)))
        print "Done!"

    if subtractRectify2:
        dirName = "a"
        path = "/Users/adamyedidia/walls/src/pole_images/legos/" + dirName + \
            "/average.p"

        print "Loading..."
        arr = pickle.load(open(path, "r"))
        print "Loaded!"

        CORNER = np.array([120, 309])
        P1 = np.array([119, 302])
        P2 = np.array([415, 193])
        OPP_CORNER = np.array([416, 200])

        print arr.shape

        rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 500, 300)

    #    viewFrame(rectifiedArr)

        dirName2 = "back"
        path2 = "/Users/adamyedidia/walls/src/pole_images/legos/" + dirName2 + \
            "/average.p"

        print "Loading..."
        arr = pickle.load(open(path2, "r"))
        print "Loaded!"

        subtractArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 500, 300)

    #    viewFrame(subtractArr)

        adjustedArr = rectifiedArr - subtractArr

        viewFrame(adjustedArr, 1e1, differenceImage=True)

        displayFlattenedFrame(flattenFrame(adjustedArr), 100, 3e1, differenceImage=True)

        path3 = "/Users/adamyedidia/walls/src/pole_images/legos/" + dirName + \
            "/rectified.p"

        matPath = "/Users/adamyedidia/walls/src/pole_images/legos/" + dirName + \
            "/arr.mat"

        print "Writing..."
        pickle.dump(flattenFrame(adjustedArr), open(path3, "w"))
        scipy.io.savemat(matPath, dict(arr=flattenFrame(adjustedArr)))
        print "Done!"


    #    viewFrame(arr)



    #    P1 = np.array([105, 88])
    #    P2 = np.array([440, 16])
    #    P1 = np.array([123, 344])
    #    CORNER = np.array([124, 352])
    #    P2 = np.array([412, 220])
    #    OPP_CORNER = np.array([411, 227])



    #    croppedArr = arr[0:100, 0:100]

    #    CORNER = np.array([0, 0])
    #    P1 = np.array([100, 0])
    #    OPP_CORNER = np.array([0, 100])
    #    P2 = np.array([100, 100])

    #    rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 500, 300)

    #    viewFrame(rectifiedArr)

    #    path2 = "/Users/adamyedidia/walls/src/pole_images/monitor_line/" + dirName + \
    #        "/rectified.p"


    #    viewFrame(rectifiedArr)

    #    print "Writing..."
    #    pickle.dump(adjustedArr, open(path2, "w"))
    #    print "Done!"

    if hallwayImaging:

        path = "/Users/adamyedidia/Dropbox (MIT)/shadowImaging/wallImaging/2017-09-13/results/loc_c.png"

        imRaw = Image.open(path)
        im = np.array(imRaw).astype(float)

        viewFrame(im)

        CORNER = np.array([195, 348])
    #    P1 = np.array([157, 57])
    #    P1 = np.array([161, 48])
        P1 = np.array([180, 27])
        OPP_CORNER = np.array([374, 1])
    #    P2 = np.array([392, 255])
        P2 = np.array([408, 248])


        rectifiedFrame = rectify(im, CORNER, P1, P2, OPP_CORNER, 200, 200)

        displayFlattenedFrame(flattenFrame(rectifiedFrame), 100)

    #    viewFrame(np.flip(rectifiedFrame, 1))

    #    path1 = "/Users/adamyedidia/Downloads/sd_card/private/m4root/THMBNL/C0112T01.JPG"

    #    imRaw = Image.open(path1)
    #    im1 = np.array(imRaw).astype(float)

    #    path2 = "/Users/adamyedidia/Downloads/sd_card/private/m4root/THMBNL/C0115T01.JPG"

    #    imRaw = Image.open(path2)
    #    im2 = np.array(imRaw).astype(float)

    #    viewFrame(-blur2DImage(im2 - im1, 8), magnification=2e1, differenceImage=False)

    if oldRectify:

        rectifiedFrame = rectify(frame, CORNER, P1, P2, OPP_CORNER, 1500, 1000)

        viewFrame(np.flip(rectifiedFrame, 1))

    if seeObs:
    #    dirName = "calibration"

    #    path = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified.p"

        dirName = "a_dark"
        path = "/Users/adamyedidia/walls/src/pole_images/monitor_line/" + dirName + \
            "/average.p"

        print "Loading..."
        arr = pickle.load(open(path, "r"))
        print "Loaded!"

        viewFrame(arr)

    if rawRectify:

    #    FILE_NAME = "/Users/adamyedidia/flags_garbled/texas_garbled/IMG_5048.CR2"
    #    arr = convertRawFileToArray(FILE_NAME)

        two = True

        dirName = "calibration"

        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/average" + ("2"*two) + \
            ".p"

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

        print arr.shape

        rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 5000, 3000)

        path2 = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified" + ("2"*two) + \
            ".p"


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
