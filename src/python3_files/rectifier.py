
import numpy as np
from PIL import Image
from video_magnifier import viewFrame, viewFrameR
from math import ceil, floor, sin, pi, cos
from cr2_processor import convertRawFileToArray
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from video_processor import processVideo, batchAndDifferentiate
from image_distortion_simulator import imageify, imageifyComplex
import pickle
import scipy.io
import sys
import imageio

radialRectify1 = False
radialRectify2 = False
oldRectify = False
rawRectify = False
wrongRectify = False
veryWrongRectify = False
seeObs = False
subtractRectify = False
subtractRectify2 = False
hallwayImaging = False
batchMovie = False
rectifyVideo = False
rectifyVideo2 = False
rectifyVideo3 = False
rectifyOrange = False
rectifyBld66 = False
rectifyBld34 = False
rectifyStata = False
rectifyFan = False
rectifyFanMonitor = False
rectifyPlant = False
rectifyPlantMonitor = False
rectifyDarpaVid = True

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

    acceptable = True

    for testIndex in [floorI, ceilI]:
        if testIndex < 0 or testIndex >= len(occluderMatrix):
            acceptable = False

    for testIndex in [floorJ, ceilJ]:
        if testIndex < 0 or testIndex >= len(occluderMatrix[0]):
            acceptable = False

    if acceptable:
        floorIfloorJ = occluderMatrix[floorI][floorJ]
        floorIceilJ = occluderMatrix[floorI][ceilJ]
        ceilIfloorJ = occluderMatrix[ceilI][floorJ]
        ceilIceilJ = occluderMatrix[ceilI][ceilJ]
    else:
#        print "Warning: fuzzyLookup index out of bounds"
        return np.array([0,0,0])

    return (1-residueI) * (1-residueJ) * floorIfloorJ + \
        (1-residueI) * residueJ * floorIceilJ + \
        residueI * (1-residueJ) * ceilIfloorJ + \
        residueI * residueJ * ceilIceilJ

def makeGrid(corner, p1, p2, oppCorner, steps1, steps2):

    returnArray = []

    for i in range(steps1):
        returnArray.append([])

#        print "Making grid:", i, "/", steps1

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

#        print "Rectifying:", j, "/", steps2

        for i in range(steps1):
            vec = grid[i][j]

            value = fuzzyLookup(arr, (vec[1], vec[0]))

            returnArray[-1].append(value)

    return np.array(returnArray)

def findIntersectionPoint(arr, p1, p2, p3, p4):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]

    mA = (y2 - y1)/(x2 - x1)
    bA = y2 - mA*x2
    bA = y1 - mA*x1

    mB = (y4 - y3)/(x4 - x3)
    bB = y4 - mB*x4
    bB = y3 - mB*x3

    print(mA, bA, mB, bB)

    x = (bB - bA)/(mA - mB) # swap intentional
#    y = mB*x + bB
    y = mA*x + bA

    return np.array([x, y])

def radialRectify(arr, p1, p2, p3, p4, MIN_Y, MAX_Y, MIN_THETA, MAX_THETA):
    circleCenter = findIntersectionPoint(arr, p1, p2, p3, p4)

#    print p1, p2, p3, p4

    print(circleCenter)

    numYTicks = 100
    numThetaTicks = 100

    returnArray = []

    for rawY in range(numYTicks):
        y = MIN_Y + (MAX_Y - MIN_Y)*rawY/numYTicks

        adjustedY = y - circleCenter[1]

        returnArray.append([])

        for rawTheta in range(0, numThetaTicks):
            theta = MIN_THETA + (MAX_THETA - MIN_THETA)*rawTheta/numThetaTicks

            r = adjustedY/sin(theta)
            x = r*cos(theta) + circleCenter[0]

            indices = (y, x) #intentional?
#            print theta, adjustedY



            val = fuzzyLookup(arr, indices)
#            print indices, val

            returnArray[-1].append(val)

    return np.swapaxes(np.array(returnArray), 0, 1)

def radialRectifyMovie(arr, p1, p2, p3, p4, MIN_Y, MAX_Y, MIN_THETA, MAX_THETA):
    returnMovie = []

    for i, frame in enumerate(arr):
        print(i, "/", len(arr))

        returnMovie.append(radialRectify(frame, p1, p2, p3, p4, MIN_Y, MAX_Y, \
            MIN_THETA, MAX_THETA))

    return returnMovie


if __name__ == "__main__":

    if radialRectify1:
        num = "272"
        path = "/Users/adamyedidia/walls/src/downsampled_ceiling" + num + ".p"

        print("Loading...")
        arr = pickle.load(open(path, "r"))
        print("Loaded!")

        p1 = np.array([14, 5])
        p2 = np.array([17, 33])
        p3 = np.array([4, 8])
        p4 = np.array([10, 31])

        MIN_Y = 0
        MAX_Y = 50

        MIN_THETA = pi
        MAX_THETA = 2*pi

        rectified = radialRectifyMovie(arr, p1, p2, p3, p4, MIN_Y, MAX_Y, MIN_THETA, MAX_THETA)

        oneDMovie = []

        for i,frame in enumerate(rectified):

            print(i)

#            batchedFrame = batchAndDifferentiate(frame, [(100, False), (1, True), (1, False)])
#            batchedFrame = batchAndDifferentiate(frame, [(1, True), (100, False), (1, False)])
#            oneDMovie.append(batchedFrame)

#        print np.array(oneDMovie)[:,0].shape

#        viewFrame(np.array(oneDMovie)[:,0], differenceImage=True, magnification=3e3)
#        viewFrame(np.array(oneDMovie)[:,:,0], differenceImage=True, magnification=1e3)

        processVideo(rectified, "00:30", [(1, False), (1, True), (1, True), (1, False)], \
            "vid_diff" + num + "_flat", magnification=1e4, firstFrame=0, lastFrame=None, toVideo=True)




#        i = 4

#        rectified = radialRectify(arr[i], p1, p2, p3, p4, MIN_Y, MAX_Y, MIN_THETA, MAX_THETA)

#        viewFrame(arr[i], magnification=10, differenceImage=False)
#        viewFrame(rectified, magnification=10, differenceImage=False)

    if radialRectify2:
        path = "/Users/adamyedidia/Dropbox (MIT)/shadowImaging/2d-imaging/data/" + \
            "2017-11-05/doorway/pointlight/point1.p"

        print("hi")

        print("Loading...")
        arr = pickle.load(open(path, "r"))
        print("Loaded!")

        viewFrame(arr)
        sys.exit()

        p1 = np.array([316, 222])
        p2 = np.array([17, 33])
        p3 = np.array([4, 8])
        p4 = np.array([10, 31])

        MIN_Y = 0
        MAX_Y = 50

        MIN_THETA = pi
        MAX_THETA = 2*pi

        rectified = radialRectifyMovie(arr, p1, p2, p3, p4, MIN_Y, MAX_Y, MIN_THETA, MAX_THETA)

        oneDMovie = []

        for i,frame in enumerate(rectified):

            print(i)

#            batchedFrame = batchAndDifferentiate(frame, [(100, False), (1, True), (1, False)])
#            batchedFrame = batchAndDifferentiate(frame, [(1, True), (100, False), (1, False)])
#            oneDMovie.append(batchedFrame)

#        print np.array(oneDMovie)[:,0].shape

#        viewFrame(np.array(oneDMovie)[:,0], differenceImage=True, magnification=3e3)
#        viewFrame(np.array(oneDMovie)[:,:,0], differenceImage=True, magnification=1e3)

        processVideo(rectified, "00:30", [(1, False), (1, True), (1, True), (1, False)], \
            "vid_diff" + num + "_flat", magnification=1e4, firstFrame=0, lastFrame=None, toVideo=True)




#        i = 4

#        rectified = radialRectify(arr[i], p1, p2, p3, p4, MIN_Y, MAX_Y, MIN_THETA, MAX_THETA)

#        viewFrame(arr[i], magnification=10, differenceImage=False)
#        viewFrame(rectified, magnification=10, differenceImage=False)



    if subtractRectify:
        dirName = "b_dark"
        path = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/" + dirName + \
            "/average.p"

        print("Loading...")
        arr = pickle.load(open(path, "r"))
        print("Loaded!")

        CORNER = np.array([120, 309])
        P1 = np.array([119, 302])
        P2 = np.array([415, 193])
        OPP_CORNER = np.array([416, 200])

        print(arr.shape)

        rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 500, 300)

    #    viewFrame(rectifiedArr)

        dirName2 = "back_dark"
        path2 = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/" + dirName2 + \
            "/average.p"

        print("Loading...")
        arr = pickle.load(open(path2, "r"))
        print("Loaded!")

        subtractArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 500, 300)

    #    viewFrame(subtractArr)

        adjustedArr = rectifiedArr - subtractArr

        viewFrame(adjustedArr, 1e1, differenceImage=True)

        displayFlattenedFrame(flattenFrame(adjustedArr), 100, 1e1, differenceImage=True)

        path3 = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/" + dirName + \
            "/rectified.p"

        matPath = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/" + dirName + \
            "/arr.mat"

        print("Writing...")
        pickle.dump(flattenFrame(adjustedArr), open(path3, "w"))
        scipy.io.savemat(matPath, dict(arr=flattenFrame(adjustedArr)))
        print("Done!")

    if subtractRectify2:
        dirName = "a"
        path = "/Users/adamyedidia/walls/src/pole_images/legos/" + dirName + \
            "/average.p"

        print("Loading...")
        arr = pickle.load(open(path, "r"))
        print("Loaded!")

        CORNER = np.array([120, 309])
        P1 = np.array([119, 302])
        P2 = np.array([415, 193])
        OPP_CORNER = np.array([416, 200])

        print(arr.shape)

        rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 500, 300)

    #    viewFrame(rectifiedArr)

        dirName2 = "back"
        path2 = "/Users/adamyedidia/walls/src/pole_images/legos/" + dirName2 + \
            "/average.p"

        print("Loading...")
        arr = pickle.load(open(path2, "r"))
        print("Loaded!")

        subtractArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 500, 300)

    #    viewFrame(subtractArr)

        adjustedArr = rectifiedArr - subtractArr

        viewFrame(adjustedArr, 1e1, differenceImage=True)

        displayFlattenedFrame(flattenFrame(adjustedArr), 100, 3e1, differenceImage=True)

        path3 = "/Users/adamyedidia/walls/src/pole_images/legos/" + dirName + \
            "/rectified.p"

        matPath = "/Users/adamyedidia/walls/src/pole_images/legos/" + dirName + \
            "/arr.mat"

        print("Writing...")
        pickle.dump(flattenFrame(adjustedArr), open(path3, "w"))
        scipy.io.savemat(matPath, dict(arr=flattenFrame(adjustedArr)))
        print("Done!")


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

        print("Loading...")
        arr = pickle.load(open(path, "r"))
        print("Loaded!")

        viewFrame(arr)

    if rawRectify:

    #    FILE_NAME = "/Users/adamyedidia/flags_garbled/texas_garbled/IMG_5048.CR2"
    #    arr = convertRawFileToArray(FILE_NAME)

        two = True

        dirName = "calibration"

        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/average" + ("2"*two) + \
            ".p"

        print("Loading...")
        arr = pickle.load(open(path, "r"))
        print("Loaded!")

        CORNER = np.array([1456, 2830])
        P2 = np.array([5130, 2916])
        OPP_CORNER = np.array([5484, 822])
        P1 = np.array([1185, 689])

    #    croppedArr = arr[0:100, 0:100]

    #    CORNER = np.array([0, 0])
    #    P1 = np.array([100, 0])
    #    OPP_CORNER = np.array([0, 100])
    #    P2 = np.array([100, 100])

        print(arr.shape)

        rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 5000, 3000)

        path2 = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified" + ("2"*two) + \
            ".p"


    #    viewFrame(rectifiedArr)

        print("Writing...")
        pickle.dump(rectifiedArr, open(path2, "w"))
        print("Done!")

    #    viewFrame(arr)

    if wrongRectify:

        dirName = "texas_garbled"

        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/average.p"

        print("Loading...")
        arr = pickle.load(open(path, "r"))
        print("Loaded!")

        CORNER = np.array([1300, 2900])
        P2 = np.array([5300, 2900])
        OPP_CORNER = np.array([5300, 700])
        P1 = np.array([1300, 700])

        rectifiedArr = rectify(arr, CORNER, P1, P2, OPP_CORNER, 5000, 3000)

        path2 = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified_wrong.p"

        print("Writing...")
        pickle.dump(rectifiedArr, open(path2, "w"))
        print("Done!")

    if veryWrongRectify:

        dirName = "texas_garbled"

        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/average.p"

        print("Loading...")
        arr = pickle.load(open(path, "r"))
        print("Loaded!")

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

        print("Writing...")
        pickle.dump(rectifiedArr, open(path2, "w"))
        print("Done!")

    if rectifyVideo:
        CORNER = np.array([100, 0])
        P1 = np.array([200, 43])
        OPP_CORNER = np.array([200, 100])
        P2 = np.array([100, 120])

        path = "/Users/adamyedidia/walls/src/cardboard.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr):
            print(i)

#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 69, 125)

#            viewFrame(rectifiedArr)

#            print i
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        pickle.dump(np.array(rectifiedVideo), open("cardboard_rect.p", "w"))

    if rectifyVideo2:

        path = "/Users/adamyedidia/walls/src/macarena_dark_fixed.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr):
            viewFrame(frame)

#            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 70, 125)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            rectifiedVideo.append(rectifiedArr)

        pickle.dump(np.array([rectifiedVideo]), open("macarena_dark_fixed_rect.p", "w"))

    if rectifyVideo3:
        CORNER = np.array([7, 43])
        P2 = np.array([58, 33])
        OPP_CORNER = np.array([57, 115])
        P1 = np.array([4, 107])

        path = "/Users/adamyedidia/walls/src/36225_bright_fixed.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr):
#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 75, 75)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        pickle.dump(np.array(rectifiedVideo), open("36225_bright_fixed_rect.p", "w"))    

    if rectifyOrange:

        # HI-RES:

        # CORNER = np.array([471, 47])
        # P2 = np.array([1182, 151])
        # OPP_CORNER = np.array([1200, 499])
        # P1 = np.array([474, 564])

        CORNER = np.array([31, 3])
        P2 = np.array([79, 10])
        OPP_CORNER = np.array([80, 33])
        P1 = np.array([31, 38])

        path = "/Users/adamyedidia/walls/src/orange.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr):
#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 75, 75)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        pickle.dump(np.array(rectifiedVideo), open("orange_rect.p", "w"))

    if rectifyBld66:
        CORNER = np.array([9, 0])
        P2 = np.array([80, 0])
        OPP_CORNER = np.array([81, 43])
        P1 = np.array([8, 42])

        path = "/Users/adamyedidia/walls/src/bld66.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr):
#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 70, 40)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

    if rectifyBld34:
        CORNER = np.array([24, 0])
        P2 = np.array([74, 8])
        OPP_CORNER = np.array([74, 37])
        P1 = np.array([24, 46])

        path = "/Users/adamyedidia/walls/src/bld34.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr):
#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 70, 40)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        pickle.dump(np.array(rectifiedVideo), open("bld34_rect.p", "w"))            

    if rectifyStata:
#        CORNER = np.array([24, 0])
 #       P2 = np.array([74, 8])
#        OPP_CORNER = np.array([74, 37])
 #       P1 = np.array([24, 46])

        CORNER = np.array([3, 53])
        P2 = np.array([63, 33])
        OPP_CORNER = np.array([71, 95])
        P1 = np.array([4, 95])

        path = "/Users/adamyedidia/walls/src/stata.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr):
#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 70, 40)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        pickle.dump(np.array(rectifiedVideo), open("stata.p", "w"))          

    if rectifyFan:
        CORNER = np.array([67, 7])
        P2 = np.array([95, 0])
        OPP_CORNER = np.array([107, 65])
        P1 = np.array([72, 60])

        path = "/Users/adamyedidia/walls/src/fan_fine.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr):
#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 70, 40)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        print(len(rectifiedVideo))

        pickle.dump(np.array(rectifiedVideo), open("fan_fine_rect.p", "w"))          

    if rectifyFanMonitor:
        CORNER = np.array([53, 10])
        P2 = np.array([95, 0])
        OPP_CORNER = np.array([107, 65])
        P1 = np.array([59, 61])

        path = "/Users/adamyedidia/walls/src/fan_monitor_fine.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr[:]):
#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 70, 40)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        pickle.dump(np.array(rectifiedVideo), open("fan_monitor_fine_rect.p", "w"))                

    if rectifyPlant:
        CORNER = np.array([67, 7])
        P2 = np.array([95, 0])
        OPP_CORNER = np.array([107, 65])
        P1 = np.array([72, 60])

        path = "/Users/adamyedidia/walls/src/plant_fine.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

        for i, frame in enumerate(arr):
#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 70, 40)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        print(len(rectifiedVideo))

        pickle.dump(np.array(rectifiedVideo), open("plant_fine_rect.p", "w"))      

    if rectifyPlantMonitor:
        CORNER = np.array([67, 7])
        P2 = np.array([95, 0])
        OPP_CORNER = np.array([107, 65])
        P1 = np.array([72, 60])

        path = "/Users/adamyedidia/walls/src/plant_monitor.p"

        arr = pickle.load(open(path, "r"))

        rectifiedVideo = []

#        print len(arr)

        for i, frame in enumerate(arr[:200]):
            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 70, 40)

            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        pickle.dump(np.array(rectifiedVideo), open("plant_monitor_rect.p", "w"))

    if rectifyDarpaVid:
#        CORNER = np.array([3, 35])
#        P2 = np.array([96, 0])
#        OPP_CORNER = np.array([88, 65])
#        P1 = np.array([1, 0])

        CORNER = np.array([3, 37])
        P2 = np.array([78, 0])
        OPP_CORNER = np.array([79, 71])
        P1 = np.array([0, 56])

        path = "/Users/adamyedidia/walls/src/darpa_vid_2.p"

        arr = pickle.load(open(path, "r"))

#        viewFrame(arr[200])

        rectifiedVideo = []

#        print len(arr)

        for i, frame in enumerate(arr):
#            viewFrame(frame)

            rectifiedArr = rectify(frame, CORNER, P1, P2, OPP_CORNER, 70, 40)

#            viewFrame(rectifiedArr)

            print(i)
#            print rectifiedArr
#            print rectifiedArr.shape

#            viewFrame(rectifiedArr)

#            viewFrame(rectifiedArr)

            rectifiedVideo.append(rectifiedArr)

        pickle.dump(np.array(rectifiedVideo), open("darpa_vid_2_rect.p", "w"))

