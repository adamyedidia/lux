from __future__ import division
import numpy as np
import pylab
from math import pi, sqrt, log, floor, exp, sin, cos
import matplotlib.pyplot as p
from PIL import Image
from PIL import ImageFilter
import random
import pickle
from process_image import ungarbleImageX, ungarbleImageY, \
    createGarbleMatrixX, createGarbleMatrixY, createGarbleMatrixFull, \
    ungarbleImageXOld, ungarbleImageYOld, getQ
from import_1dify import batchList, displayConcatenatedArray, fuzzyLookup, stretchArray, fuzzyLookup2D, \
    getTransferMatrixFromOccluder
import imageio
from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import sys
from scipy.linalg import dft
from best_matrix import findAssignment, isPrime, xorProduct, padIntegerWithZeros, \
    getToeplitzLikeTransferMatrixWithVariedDepth
from scipy.linalg import circulant, toeplitz
from scipy.io import loadmat
#from video_processor import batchArrayAlongAxis, batchAndDifferentiate, convertArrayToVideo

MATRIX_SIZE_TEST = False
FOURIER_UNDERSTANDER = False
FOURIER_UNDERSTANDER_2D = False
ALRIGHT_LETS_DO_THIS = False
TOEPLITZ = False
TOEPLITZ_2D = False
MAKE_ELEPHANT = False
ANTONIO_METHOD = False
DIFFERENT_DEPTHS_SIM = False
ACTIVE_SIM = False
ANGLE_CORNER = False
LIGHT_OF_THE_SUN_SIM = False
SPHERE_DISK_COMPARISON = False
DISPERSION_PLOT = False
LOAD_BU_SIM_DATA = False
LOAD_BU_REAL_DATA = False
JOIN_INFO = False
JOIN_INFO_BY_LIST = False
PROJECTION_INFO = False
EQUIV_OCC = True
EQUIV_OCC_2 = False
EQUIV_OCC_3 = False
EQUIV_OCC_MOVIE_1 = False
EQUIV_OCC_MOVIE_2 = False

def hasOptimalCirculant(x):
    if int(log(x+1, 2)) == log(x+1, 2):
        return True
    if isPrime(x) and x % 4 == 3:
        return True
    if x in [15, 35, 143, 323, 899, 1763, 3599]:
        # twin prime products
        return True

    return False

def upperTriangular(n):
    return np.triu(np.ones((n, n)))

def spectrallyFlatOccluder(imShape):
    highVal = 1/(imShape[0]*imShape[1])

    if hasOptimalCirculant(imShape[0]):
        if hasOptimalCirculant(imShape[1]):

            ass0 = findAssignment(imShape[0])[1]
            ass1 = findAssignment(imShape[1])[1]

            print np.array(xorProduct(ass0, ass1))

            occluder = highVal * np.array(xorProduct(ass0, ass1))

            return occluder

    raise

# elementwise kron with (255, 255, 255)
def imageify(arr):
    result = np.reshape(np.kron(arr, np.array([255,255,255])), arr.shape + tuple([3]))

    return result

def imageifyComplex(arr):
    result = np.reshape(np.kron(np.real(arr), np.array([0,0,255])), arr.shape + tuple([3])) + \
        np.reshape(np.kron(np.imag(arr), np.array([255,0,0])), arr.shape + tuple([3]))

    return result

def batchArrayAlongZeroAxis(arr, batchSize):

    listOfBigFrames = []
    frameSum = np.zeros(arr[0].shape)
    numFrames = len(arr)

    for i in range(numFrames):
        frameSum += arr[i]

        if i % batchSize == batchSize - 1:
            listOfBigFrames.append(frameSum / batchSize)
            frameSum = np.zeros(arr[0].shape)

    if numFrames % batchSize != 0:
        listOfBigFrames.append(frameSum / (numFrames % batchSize))

#    print len(listOfBigFrames)
    return np.array(listOfBigFrames)

def batchArrayAlongAxis(arr, axis, batchSize):
    rearrangedArr = np.swapaxes(arr, 0, axis)
    batchedArray = batchArrayAlongZeroAxis(rearrangedArr, batchSize)
    return(np.swapaxes(batchedArray, 0, axis))

def batchAndDifferentiate(arr, listOfResponses):
    dim  = len(arr.shape)

    assert dim == len(listOfResponses)

    # batch things
#    print "Batching..."
    for i in range(dim):
        arr = batchArrayAlongAxis(arr, i, listOfResponses[i][0])

#    viewFrame(arr, 1e0, False)

    # take gradients
#    print "Differentiating..."
    for i in range(dim - 1, -1, -1):
        if listOfResponses[i][1]:
            arr = np.gradient(arr, axis=i)

#            viewFrame(arr, 3e2, True)

#    arr = blur2DImage(arr, 5)

#    viewFrame(arr, 3e2, True)

    return arr


def makeCornerTransferMatrix(vector1, vector2, imShape, vertRadius, horizRadius):

    planeOfOcclusion = np.cross(vector1, vector2)

    maxX = imShape[0]
    maxY = imShape[1]

    returnMat = []

    for q, sceneX in enumerate(np.linspace(-horizRadius, horizRadius, maxX)):
        for p, sceneY in enumerate(np.linspace(-horizRadius, horizRadius, maxY)):
            print sceneX, maxX

            scenePoint = np.array([sceneX, sceneY, vertRadius])
            returnMat.append([])

            for obsX in np.linspace(-horizRadius, horizRadius, maxX):

                for obsY in np.linspace(-horizRadius, horizRadius, maxY):
                    obsPoint = np.array([obsX, obsY, -vertRadius])

                    a = planeOfOcclusion[0]
                    b = planeOfOcclusion[1]
                    c = planeOfOcclusion[2]

                    d = sceneX
                    e = sceneY
                    f = vertRadius

                    g = obsX
                    h = obsY
                    i = -vertRadius

                    t = (a*d + b*e + c*f)/(a*d + b*e + c*f - a*g - b*h - c*i)

                    intX = d + (g-d)*t
                    intY = e + (h-e)*t
                    intZ = f + (i-f)*t

                    newBasis = np.linalg.inv(np.transpose(np.array([vector1, vector2, np.array([0,0,1])])))

                    intVec = np.dot(newBasis, np.array([intX, intY, intZ]))

    #                print intVec
                    closestToZero = min(intVec[0], intVec[1])

                    # this is a crappy increment
                    increment = horizRadius / max(maxX, maxY)

                    if closestToZero > increment/2:
                        returnMat[-1].append(1/(maxX*maxY))
                    elif closestToZero < -increment/2:
                        returnMat[-1].append(0)
                    else:
                        returnMat[-1].append(1/(maxX*maxY) * (closestToZero + increment/2)/increment)

            viewFrame(vectorToImage(imageify(np.array(returnMat[-1])), imShape), 1e2, \
                filename="anglemovie/img_" + padIntegerWithZeros(p*maxX + q, 3) + ".png")

    return np.transpose(np.array(returnMat))

def oneDPinHole(length, widthOfPinhole):
    return [1/length*(abs(i - length/2) < length*widthOfPinhole/2) for i in range(length)]

def oneDRandom(length):
    return [1/length*(random.random() > 0.5) for _ in range(length)]

def oneDedge(length):
    return [0]*int((length-1)/2)+[1/length]*int((length+1)/2)

def oneDNothing(length):
    return [1/length for _ in range(length)]

def horizontalColumn(imShape, height, highVal=None):
    print "column"

    midPointX = int(imShape[0]/2)

    if highVal == None:
        highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if abs(i - midPointX) < height*imShape[0]:
                returnArray[i][j] = highVal

#    p.matshow(returnArray)
#    p.show()
    return returnArray

def convolve2DToeplitz(arr1, arr2):
    return convolve2d(arr1, arr2, mode="valid")

def resizeArray(arr, newShape):
    returnArray = []

    xs = np.linspace(0, arr.shape[0]-1, newShape[0])
    ys = np.linspace(0, arr.shape[1]-1, newShape[1])

    for x in xs:
        returnArray.append([])
        for y in ys:
            returnArray[-1].append(fuzzyLookup2D(arr, x, y))

    returnArray = np.array(returnArray)

#    print returnArray.shape

    return returnArray

def getR(im):
    return np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)[0]

def padArrayToShape(arr, shape, padVal=0):
    arrShape = arr.shape

    diff0 = shape[0] - arrShape[0]
    diff1 = shape[1] - arrShape[1]

    diff0above = int(diff0/2)
    diff0below = diff0 - diff0above

    diff1above = int(diff1/2)
    diff1below = diff1 - diff1above

    return np.pad(arr, [(diff0above, diff0below), (diff1above, diff1below)], "constant",
        constant_values=padVal)


def elephant(imShape, width):
    elephantArr = pickle.load(open("elephant.p", "r"))

    elephantBW = getR(elephantArr)

    highVal = 1/(imShape[0]*imShape[1])

    elephantShape = [width*i for i in imShape]

    innerElephant = resizeArray(elephantBW, elephantShape)

    def moreThan127(x):
        return 1*(x > 127)

    innerElephantThresholded = np.vectorize(moreThan127)(innerElephant)

    viewFrame(innerElephantThresholded)

    innerElephantPadded = padArrayToShape(innerElephantThresholded, imShape, padVal=1)

    viewFrame(innerElephantPadded)

    return highVal*innerElephantPadded

def elephantOccluder(imShape, width):
    elephantArr = pickle.load(open("elephant.p", "r"))

    elephantBW = getR(elephantArr)

    highVal = 1/(imShape[0]*imShape[1])

    elephantShape = [width*i for i in imShape]

    innerElephant = resizeArray(elephantBW, elephantShape)

    def lessThan127(x):
        return 1*(x <= 127)

    innerElephantThresholded = np.vectorize(lessThan127)(innerElephant)

    innerElephantPadded = padArrayToShape(innerElephantThresholded, imShape, padVal=0)

    return highVal*innerElephantPadded

def verticalColumn(imShape, width, highVal=None):
    print "column"

    midPointY = int(imShape[1]/2)

    if highVal == None:
        highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if abs(j - midPointY) < width*imShape[1]:
                returnArray[i][j] = highVal

#    p.matshow(returnArray)
#    p.show()
    return returnArray

def lens(imShape):
    highVal = 1
    # This is 1 on purpose!!

    returnArray = np.zeros(imShape)

    returnArray[-1][-1] = highVal

    return returnArray

def lens2(imShape):
    print "pinhole"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1
    # This is 1 on purpose!!

    returnArray = np.zeros(imShape)

    returnArray[midPointX][midPointY] = highVal

    return returnArray

def pinSquare(imShape, squareRadius):
    print "square"

    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if abs(i - midPointX) < squareRadius*imShape[0]:
                if abs(j - midPointY) < squareRadius*imShape[0]:
                    returnArray[i][j] = highVal

#    p.matshow(returnArray)
#    p.show()
    return returnArray

def occluderGenerator(derivativeArray, verticalConstants, horizontalConstants):
    arrShape = derivativeArray.shape

 #   viewFrame(imageify(derivativeArray), adaptiveScaling=True, differenceImage=True)

    occluderVerticaled = np.cumsum(derivativeArray, 0)

#    viewFrame(imageify(occluderVerticaled), adaptiveScaling=True, differenceImage=True)

    occluderVerticalConstantsAdd = np.tile(verticalConstants, (arrShape[0], 1))

 #   viewFrame(imageify(occluderVerticalConstantsAdd), adaptiveScaling=True, \
 #       differenceImage=True)

    occluderVerticaledWithConstants = occluderVerticaled + \
        occluderVerticalConstantsAdd

 #   viewFrame(imageify(occluderVerticaledWithConstants), adaptiveScaling=True, \
 #       differenceImage=True)

    occluderHorizontaled = np.cumsum(occluderVerticaledWithConstants, 1)

    print "1"

    viewFrame(imageify(occluderHorizontaled), adaptiveScaling=True, \
        differenceImage=True)

    occluderHorizontalConstantsAdd = np.transpose(np.tile(horizontalConstants, \
        (arrShape[1], 1)))

    print occluderHorizontalConstantsAdd

    print "2"
    viewFrame(imageify(occluderHorizontalConstantsAdd), adaptiveScaling=True, \
        differenceImage=True)

    occluder = occluderHorizontaled + occluderHorizontalConstantsAdd

    print "3"
    viewFrame(imageify(occluder), adaptiveScaling=True, \
        differenceImage=True)
    
    return occluder

def twoSquares(imShape, squareRadius):
    print "square"

    midPointX1 = int(imShape[0]/4)
    midPointY1 = int(imShape[1]/4)

    midPointX2 = int(3*imShape[0]/4)
    midPointY2 = int(3*imShape[1]/4)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if abs(i - midPointX1) < squareRadius*imShape[0]:
                if abs(j - midPointY1) < squareRadius*imShape[0]:
                    returnArray[i][j] = highVal

            if abs(i - midPointX2) < squareRadius*imShape[0]:
                if abs(j - midPointY2) < squareRadius*imShape[0]:
                    returnArray[i][j] = highVal

#    p.matshow(returnArray)
#    p.show()
    return returnArray


def squareSpeck(imShape, squareRadius):
    print "square"

    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = highVal*np.ones(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if abs(i - midPointX) < squareRadius*imShape[0]:
                if abs(j - midPointY) < squareRadius*imShape[0]:
                    returnArray[i][j] = 0

#    p.matshow(returnArray)
#    p.show()
    return returnArray

def vectorizedDot(mat, arr, targetShape):
    return np.reshape(np.dot(mat, arr.flatten()), targetShape)

def pinCircle(imShape, radius, highVal=None):
    print "circle"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    if highVal == None:
        highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if sqrt((i - midPointX)**2 + (j - midPointY)**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = highVal

#    p.matshow(returnArray)
#    p.show()

    return returnArray

def cornerCircle(imShape, radius):
    print "corner circle"

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if sqrt(i**2 + j**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = highVal
            if sqrt((imShape[0]-i)**2 + j**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = highVal
            if sqrt(i**2 + (imShape[1]-j)**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = highVal
            if sqrt((imShape[0]-i)**2 + (imShape[1]-j)**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = highVal

#    p.matshow(returnArray)
#    p.show()

    return returnArray

def cornerSpeck(imShape, radius):
    print "corner circle"

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = highVal*np.ones(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if sqrt(i**2 + j**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = 0
            if sqrt((imShape[0]-i)**2 + j**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = 0
            if sqrt(i**2 + (imShape[1]-j)**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = 0
            if sqrt((imShape[0]-i)**2 + (imShape[1]-j)**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = 0

#    p.matshow(returnArray)
#    p.show()

    return returnArray

def circleSpeck(imShape, radius):
    print "circle"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = highVal*np.ones(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if sqrt((i - midPointX)**2 + (j - midPointY)**2) < radius*imShape[0]*4/pi:
                returnArray[i][j] = 0

#    p.matshow(returnArray)
#    p.show()

    return returnArray


def pinHole(imShape):
    print "pinhole"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)

    returnArray[midPointX][midPointY] = highVal

    return returnArray

def pinSpeck(imShape):
    print "pinspeck"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = highVal*np.ones(imShape)

    returnArray[midPointX][midPointY] = 0

    return returnArray

def coarseCheckerBoard(imShape):
    print "coarse checkerboard"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if i < midPointX:
                if j < midPointY:
                    returnArray[i][j] = highVal
            if i > midPointX:
                if j > midPointY:
                    returnArray[i][j] = highVal

    return returnArray

def fineCheckerBoard(imShape):
    print "fine checkerboard"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if (i+j) % 2 == 0:
                returnArray[i][j] = highVal

    return returnArray

def cornerOccluder(imShape):
    print "corner occluder"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if (i > midPointX) or (j > midPointY):
                returnArray[i][j] = highVal
            else:
                returnArray[i][j] = 0

    return returnArray

def cornerOccluderMovable(topLeftX, topLeftY, imShape):
    print "corner occluder"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if (i > midPointX + topLeftX) or (j > midPointY + topLeftY) or \
                (i < topLeftX) or (j < topLeftY):
                returnArray[i][j] = highVal
            else:
                returnArray[i][j] = 0

    return returnArray

def cornerOccluderMovingCenter(cornerLoc, imShape):
    print "corner occluder"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if (i > cornerLoc[0]) or (j > cornerLoc[1]):
                returnArray[i][j] = highVal
            else:
                returnArray[i][j] = 0

    return returnArray

def cornerCheckerboard(imShape, reverse=False):
    print "corner occluder"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if reverse:
                if (i > midPointX) and (j > midPointY):
                    returnArray[i][j] = 0
                elif (i <= midPointX) and (j <= midPointY):
                    returnArray[i][j] = 0                
                else:
                    returnArray[i][j] = highVal

            else:
                if (i > midPointX) and (j > midPointY):
                    returnArray[i][j] = highVal
                elif (i <= midPointX) and (j <= midPointY):
                    returnArray[i][j] = highVal                
                else:
                    returnArray[i][j] = 0

    return returnArray


def cornerAperture(imShape):
    print "corner aperture"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = np.zeros(imShape)
    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if (i > midPointX) and (j > midPointY):
                returnArray[i][j] = highVal
            else:
                returnArray[i][j] = 0

    return returnArray


def randomOccluder(imShape):
    print "random occluder"

    highVal = 1/(imShape[0]*imShape[1])

    returnArray = highVal*np.random.randint(0, high=2, size=imShape)
    return returnArray

def imageToVector(rgbIm):
    rearrangedIm = np.swapaxes(np.swapaxes(rgbIm, 0, 2), 1, 2)
    rIm = rearrangedIm[0]
    gIm = rearrangedIm[1]
    bIm = rearrangedIm[2]

    flatR = rIm.flatten()
    flatG = gIm.flatten()
    flatB = bIm.flatten()

    resultVec = np.swapaxes(np.array([flatR, flatG, flatB]), 0, 1)

    return resultVec

def vectorToImage(rgbVec, imSpatialDimensions):
    rearrangedVec = np.swapaxes(rgbVec, 0, 1)
    rIm = rearrangedVec[0]
    gIm = rearrangedVec[1]
    bIm = rearrangedVec[2]

    arrR = rIm.reshape(imSpatialDimensions)
    arrG = gIm.reshape(imSpatialDimensions)
    arrB = bIm.reshape(imSpatialDimensions)

    resultIm = np.swapaxes(np.swapaxes(np.array([arrR, arrG, \
        arrB]), 1, 2), 0, 2)

    return resultIm

def flattenRGB(rgbIm):
    rearrangedIm = np.swapaxes(np.swapaxes(rgbIm, 0, 2), 1, 2)
    rIm = rearrangedIm[0]
    gIm = rearrangedIm[1]
    bIm = rearrangedIm[2]

    flatR = rIm.flatten()
    flatG = gIm.flatten()
    flatB = bIm.flatten()

    resultIm = np.swapaxes(np.array([flatR, flatG, \
        flatB]), 0, 1)

    return resultIm

def doFuncToEachChannel(func, rgbIm):
    rearrangedIm = np.swapaxes(np.swapaxes(rgbIm, 0, 2), 1, 2)
    rIm = rearrangedIm[0]
    gIm = rearrangedIm[1]
    bIm = rearrangedIm[2]

    funkyR = func(rIm)
    funkyG = func(gIm)
    funkyB = func(bIm)

    resultIm = np.swapaxes(np.swapaxes(np.array([funkyR, funkyG, \
        funkyB]), 1, 2), 0, 2)

    return resultIm

def separate(rgbIm):
    rearrangedIm = np.swapaxes(np.swapaxes(rgbIm, 0, 2), 1, 2)
    
    return rearrangedIm

def doFuncToEachChannelSeparated(func, rgbIm):
    rearrangedIm = np.swapaxes(np.swapaxes(rgbIm, 0, 2), 1, 2)
    rIm = rearrangedIm[0]
    gIm = rearrangedIm[1]
    bIm = rearrangedIm[2]

    funkyR = func(rIm)
    funkyG = func(gIm)
    funkyB = func(bIm)

    return np.array([funkyR, funkyG, funkyB])

def doFuncToEachChannelTwoInputs(func, rgbIm1, rgbIm2):
    rearrangedIm1 = np.swapaxes(np.swapaxes(rgbIm1, 0, 2), 1, 2)
    rearrangedIm2 = np.swapaxes(np.swapaxes(rgbIm2, 0, 2), 1, 2)


    rIm1 = rearrangedIm1[0]
    gIm1 = rearrangedIm1[1]
    bIm1 = rearrangedIm1[2]

    rIm2 = rearrangedIm2[0]
    gIm2 = rearrangedIm2[1]
    bIm2 = rearrangedIm2[2]

    funkyR = func(rIm1, rIm2)
    funkyG = func(gIm1, gIm2)
    funkyB = func(bIm1, gIm2)

    resultIm = np.swapaxes(np.swapaxes(np.array([funkyR, funkyG, \
        funkyB]), 1, 2), 0, 2)

    return resultIm


def doFuncToEachChannelSeparatedTwoInputs(func, rgbIm1, rgbIm2):
    rearrangedIm1 = np.swapaxes(np.swapaxes(rgbIm1, 0, 2), 1, 2)
    rearrangedIm2 = np.swapaxes(np.swapaxes(rgbIm2, 0, 2), 1, 2)


    rIm1 = rearrangedIm1[0]
    gIm1 = rearrangedIm1[1]
    bIm1 = rearrangedIm1[2]

    rIm2 = rearrangedIm2[0]
    gIm2 = rearrangedIm2[1]
    bIm2 = rearrangedIm2[2]

    funkyR = func(rIm1, rIm2)
    funkyG = func(gIm1, gIm2)
    funkyB = func(bIm1, gIm2)

    return np.array([funkyR, funkyG, funkyB])

def doSeparateFuncToEachChannel(separateFunc, arr):
    funkyR = separateFunc[0](arr)
    funkyG = separateFunc[1](arr)
    funkyB = separateFunc[2](arr)

#    viewFrame(imageify(funkyR), adaptiveScaling=True)
#    viewFrame(imageify(funkyG), adaptiveScaling=True)
#    viewFrame(imageify(funkyB), adaptiveScaling=True)


    resultIm = np.swapaxes(np.swapaxes(np.array([funkyR, funkyG, \
        funkyB]), 1, 2), 0, 2)

    return resultIm

def doSeparateFuncToEachChannelSeparated(separateFunc, rgbIm):
    rearrangedIm = np.swapaxes(np.swapaxes(rgbIm, 0, 2), 1, 2)
    rIm = rearrangedIm[0]
    gIm = rearrangedIm[1]
    bIm = rearrangedIm[2]

    funkyR = separateFunc[0](rIm)
    funkyG = separateFunc[1](gIm)
    funkyB = separateFunc[2](bIm)

    resultIm = np.swapaxes(np.swapaxes(np.array([funkyR, funkyG, \
        funkyB]), 1, 2), 0, 2)

    return resultIm

def doSeparateFuncToEachChannelActuallySeparated(separateFunc, rgbIm):
    rearrangedIm = np.swapaxes(np.swapaxes(rgbIm, 0, 2), 1, 2)
    rIm = rearrangedIm[0]
    gIm = rearrangedIm[1]
    bIm = rearrangedIm[2]

    funkyR = separateFunc[0](rIm)
    funkyG = separateFunc[1](gIm)
    funkyB = separateFunc[2](bIm)

    return np.array([funkyR, funkyG, funkyB])

def swapChannels(arr, channel1, channel2):
    rearrangedIm = np.swapaxes(np.swapaxes(arr, 0, 2), 1, 2)

    newIm = []

    for i in range(3):
        if i == channel1:
            newIm.append(rearrangedIm[channel2])
        elif i == channel2:
            newIm.append(rearrangedIm[channel1])
        else:
            newIm.append(rearrangedIm[i])

    resultIm = np.swapaxes(np.swapaxes(np.array(newIm), 1, 2), 0, 2)

    return resultIm

def doFuncToEachChannelVec(func, rgbVec):
    rearrangedVec = np.swapaxes(rgbVec, 0, 1)
    rVec = rearrangedVec[0]
    gVec = rearrangedVec[1]
    bVec = rearrangedVec[2]

    funkyR = func(rVec)
    funkyG = func(gVec)
    funkyB = func(bVec)

    resultVec = np.swapaxes(np.array([funkyR, funkyG, \
        funkyB]), 0, 1)

    return resultVec

def makeImpulseImage(imShape, location="center"):
    returnArray = np.zeros((imShape[0], imShape[1], 3))

    if location == "center":
        returnArray[int(imShape[0]/2)][int(imShape[1]/2)][:] = 255

    if location == "upper-left":
        returnArray[0][0][:] = 255

    viewFrame(returnArray)

    return returnArray

def makeSquareImage(imShape, squareRadius):
    returnArray = np.zeros((imShape[0], imShape[1], 3))

    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

    for i in range(imShape[0]):
        for j in range(imShape[1]):
            if abs(i - midPointX) < squareRadius*imShape[0]:
                if abs(j - midPointY) < squareRadius*imShape[0]:
                    returnArray[i][j][:] = 255

    return returnArray

# better hope x isn't a small negative number!
def oneOverXSoft(x):
    if x == 0:
        return 1e10
    else:
        return 1/x

def convolve2dMaker(convolvant):
    def convolve2(im):
        convolutionResult = convolve2d(im, convolvant, boundary="wrap", mode="full")

        clippedConvolutionResult = convolutionResult[:im.shape[0], :im.shape[1]]

        return clippedConvolutionResult
    return convolve2

def convolve2dMakerToeplitz(convolvant):
    def convolve2(im):
        convolutionResult = convolve2d(im, convolvant, boundary="fill", mode="same")

#        print convolutionResult

#        clippedConvolutionResult = convolutionResult[:im.shape[0], :im.shape[1]]
        clippedConvolutionResult = convolutionResult

        return clippedConvolutionResult
    return convolve2

def addNoise(obsPlaneIm, w):
    thermalNoise = np.random.normal(0, w, obsPlaneIm.shape)
    shotNoise = np.random.normal(0, np.sqrt(obsPlaneIm), obsPlaneIm.shape)

#    print thermalNoise

    return obsPlaneIm + thermalNoise + shotNoise

# HACK
def addExpNoise(obsPlaneIm, w):

    print obsPlaneIm.shape[0]

    thermalNoise = np.random.exponential(w*np.sqrt([i[0] for i in obsPlaneIm]), obsPlaneIm.shape[0])

    thermalNoise = np.swapaxes(np.array([thermalNoise]*3), 0, 1)

    print obsPlaneIm.shape
    print thermalNoise.shape

    return obsPlaneIm + thermalNoise

def getRecoveryWindowSimple(occluderWindow):
    occluderWindowFrequencies = np.fft.fft2(occluderWindow)

    print occluderWindowFrequencies

    oneOverXVectorized = np.vectorize(oneOverXSoft)

    occluderWindowFrequenciesInverted = oneOverXVectorized(occluderWindowFrequencies)

    recoveryWindow = np.fft.ifft2(occluderWindowFrequenciesInverted)

#    print recoveryWindow

#    p.matshow(np.real(recoveryWindow))
#    print "recovery window: lower left"
#    p.show()

    return recoveryWindow

def getAttenuationMatrix(occluderWindowShape, beta):
    returnArray = []

    maxX = occluderWindowShape[0]
    maxY = occluderWindowShape[1]

    hypotenuse = sqrt(maxX*maxX + maxY*maxY)

    for i in range(occluderWindowShape[0]):
        returnArray.append([])

        for j in range(occluderWindowShape[1]):
            returnArray[-1].append(beta**(sqrt(i*i + j*j)/hypotenuse))

    return np.array(returnArray)

def getAttenuationMatrixOneOverF(occluderWindowShape):
    returnArray = []

    maxX = occluderWindowShape[0]
    maxY = occluderWindowShape[1]

    hypotenuse = sqrt(maxX*maxX + maxY*maxY)

    for i in range(occluderWindowShape[0]):
        returnArray.append([])

        for j in range(occluderWindowShape[1]):
            returnArray[-1].append(1/(1 + sqrt(i*i + j*j)/hypotenuse))

    return np.array(returnArray)

def getRecoveryWindowSophisticated(occluderWindow, beta, snr):
    occluderWindowFrequencies = np.fft.fft2(occluderWindow)

    attenMat = getAttenuationMatrix(occluderWindow.shape, beta)

#    p.matshow(attenMat)
#    p.colorbar()
#    p.show()


    oneOverXVectorized = np.vectorize(lambda x: 1/x)

    modifiedFreqs = snr*abs(np.multiply(occluderWindowFrequencies, occluderWindowFrequencies))

#    print modifiedFreqs
#    print "a"
#    p.matshow(np.log(np.abs(modifiedFreqs)))
#    p.colorbar()
#    p.show()

    modifiedFreqs = np.multiply(modifiedFreqs, attenMat)
#    modifiedFreqs = np.abs(np.multiply(modifiedFreqs, attenMat))

#    print modifiedFreqs

#    print "b"
#    p.matshow(np.log(np.abs(modifiedFreqs)))
#    p.colorbar()
#    p.show()

    modifiedFreqs += np.ones(occluderWindow.shape)

#    print modifiedFreqs

#    print "c"
#    p.matshow(np.log(np.abs(modifiedFreqs)))
#    p.colorbar()
#    p.show()

    occluderWindowFrequenciesDoubleInverted = oneOverXVectorized(modifiedFreqs)

#    print occluderWindowFrequenciesDoubleInverted


#    print "d"
#    p.matshow(np.log(np.abs(occluderWindowFrequenciesDoubleInverted)))
#    p.colorbar()
#    p.show()

    occluderWindowFrequenciesInverted = snr*np.multiply(occluderWindowFrequenciesDoubleInverted, \
        np.conj(occluderWindowFrequencies))

#    print "e"
#    p.matshow(np.log(np.abs(occluderWindowFrequenciesInverted)))
#    p.colorbar()
#    p.show()

    recoveryWindow = np.fft.ifft2(occluderWindowFrequenciesInverted)

#    print recoveryWindow

#    p.matshow(np.real(recoveryWindow))
#    p.colorbar()
#    print "recovery window: lower left"
#    p.show()

    return recoveryWindow*1.3

def getRecoveryWindowOneOverF(occluderWindow, snr):
    occluderWindowFrequencies = np.fft.fft2(occluderWindow)

    attenMat = getAttenuationMatrixOneOverF(occluderWindow.shape)

#    p.matshow(attenMat)
#    p.colorbar()
#    p.show()


    oneOverXVectorized = np.vectorize(lambda x: 1/x)

    modifiedFreqs = snr*abs(np.multiply(occluderWindowFrequencies, occluderWindowFrequencies))

#    print modifiedFreqs
#    print "a"
#    p.matshow(np.log(np.abs(modifiedFreqs)))
#    p.colorbar()
#    p.show()

    modifiedFreqs = np.multiply(modifiedFreqs, attenMat)
#    modifiedFreqs = np.abs(np.multiply(modifiedFreqs, attenMat))

#    print modifiedFreqs

#    print "b"
#    p.matshow(np.log(np.abs(modifiedFreqs)))
#    p.colorbar()
#    p.show()

    modifiedFreqs += np.ones(occluderWindow.shape)

#    print modifiedFreqs

#    print "c"
#    p.matshow(np.log(np.abs(modifiedFreqs)))
#    p.colorbar()
#    p.show()

    occluderWindowFrequenciesDoubleInverted = oneOverXVectorized(modifiedFreqs)

#    print occluderWindowFrequenciesDoubleInverted


#    print "d"
#    p.matshow(np.log(np.abs(occluderWindowFrequenciesDoubleInverted)))
#    p.colorbar()
#    p.show()

    occluderWindowFrequenciesInverted = snr*np.multiply(occluderWindowFrequenciesDoubleInverted, \
        np.conj(occluderWindowFrequencies))

#    print "e"
#    p.matshow(np.log(np.abs(occluderWindowFrequenciesInverted)))
#    p.colorbar()
#    p.show()

    recoveryWindow = np.fft.ifft2(occluderWindowFrequenciesInverted)

#    print recoveryWindow

#    p.matshow(np.real(recoveryWindow))
#    p.colorbar()
#    print "recovery window: lower left"
#    p.show()

    return recoveryWindow*1.3


def averageVertically(im):
    listOfSums = []

    for j in range(im.shape[1]):
        listOfSums.append(np.array([0.,0.,0.]))

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
#            print listOfSums[j], im[i][j]
            listOfSums[j] += im[i][j]

    print (np.array(im.shape[0]*[listOfSums])/im.shape[0]).shape
    return np.array(im.shape[0]*[listOfSums])/im.shape[0]


def getBitsOfOccluder(occluderWindow, beta, snr):

    occluderWindowFrequencies = np.fft.fft2(occluderWindow)

    attenMat = getAttenuationMatrix(occluderWindow.shape, beta)
    modifiedFreqs = snr*np.multiply(occluderWindowFrequencies, occluderWindowFrequencies)
    modifiedFreqs = np.abs(np.multiply(modifiedFreqs, attenMat))
    modifiedFreqs += np.ones(occluderWindow.shape)

#    p.matshow(np.abs(np.log2(modifiedFreqs)))
#    print np.abs(np.log2(modifiedFreqs))


#    print "eig contributions: upper right"
#    p.colorbar()
#    p.show()

    return np.sum(np.sum(abs(np.log2(modifiedFreqs)), 0), 0)

def viewRepeatedOccluder(occluderWindow, filename=None):
    doubleOccluder = np.concatenate((occluderWindow, occluderWindow), 0)
    quadrupleOccluder = np.concatenate((doubleOccluder, doubleOccluder), 1)

    p.matshow(quadrupleOccluder, cmap='gray')
    print "occluder: upper left"

    if filename == None:
        p.show()
    else:
        p.savefig(filename)

def viewSingleOccluder(occluderWindow, filename=None):
    p.matshow(occluderWindow, cmap='gray')
    print "single occluder: upper left"

    if filename == None:
        p.show()
    else:
        p.savefig(filename)

def buildActiveTransferMatrixNoCorner(illuminationX, sceneXRes, sceneYRes, \
    obsXRes, obsTRes, y1, y2, x1):

    startTime = 2*y2
    endTime = 2*sqrt(x1**2 + (y1+y2)**2)

    totalTime = endTime - startTime
    timeBinSize = totalTime / obsTRes

    returnMat = []

    for i, sceneX in enumerate(np.linspace(0, x1, sceneXRes)):
        for j, sceneY in enumerate(np.linspace(0, y1, sceneYRes)):

            sceneCoord = np.array([sceneX, sceneY])
            illuminationCoord = np.array([illuminationX, y1+y2])

            illuminationDistance = np.linalg.norm(sceneCoord - illuminationCoord)

            matColumn = []

            for k, obsX in enumerate(np.linspace(0, x1, obsXRes)):

                measurementCoord = np.array([obsX, y1+y2])
                measurementDistance = np.linalg.norm(sceneCoord - measurementCoord)
                totalTime = measurementDistance + illuminationDistance

                for t, obsT in enumerate(np.linspace(startTime+timeBinSize/2, \
                    endTime-timeBinSize/2, obsTRes)):

#                    print obsT, illuminationDistance, measurementDistance, timeBinSize / 2

                    k = 1

                    attenuationFactor = 1/(k+measurementDistance**2)

                    if abs(obsT-totalTime) < timeBinSize / 2:
                        matColumn.append(attenuationFactor)
                    else:
                        matColumn.append(0)

#            print matColumn


            returnMat.append(matColumn)

    return np.transpose(np.array(returnMat))

def buildActiveTransferMatrixWithEdge(illuminationX, sceneXRes, sceneYRes, \
    obsXRes, obsTRes, y1, y2, x1):

    startTime = 2*y2
    endTime = 2*sqrt(x1**2 + (y1+y2)**2)

    totalTime = endTime - startTime
    timeBinSize = totalTime / obsTRes

    returnMat = []

    r = 1

#    for j, sceneY in enumerate(np.linspace(0, y1, sceneYRes)):
#        for i, sceneX in enumerate(np.linspace(0, x1, sceneXRes)):
    for i, sceneX in enumerate(np.linspace(0, x1, sceneXRes)):
        for j, sceneY in enumerate(np.linspace(0, y1, sceneYRes)):

            sceneCoord = np.array([sceneX, sceneY])
            illuminationCoord = np.array([illuminationX, y1+y2])

            illuminationDistance = np.linalg.norm(sceneCoord - illuminationCoord)

            matColumn = []

            for k, obsX in enumerate(np.linspace(0, x1, obsXRes)):

                measurementCoord = np.array([obsX, y1+y2])
                measurementDistance = np.linalg.norm(sceneCoord - measurementCoord)
                totalTime = measurementDistance + illuminationDistance

                occlusionCoord = obsX*(y2*(1-r) + y1-sceneY) + sceneX*(y2*r)

                for t, obsT in enumerate(np.linspace(startTime+timeBinSize/2, \
                    endTime-timeBinSize/2, obsTRes)):

#                    print obsT, illuminationDistance, measurementDistance, timeBinSize / 2

                    if (abs(obsT-totalTime) < timeBinSize / 2) and \
                        (occlusionCoord > x1/2):

                        matColumn.append(1)
                    else:
                        matColumn.append(0)

#            print matColumn


            returnMat.append(matColumn)

    return np.transpose(np.array(returnMat))

def buildActiveTransferMatrixWithOccluder(illuminationX, sceneXRes, sceneYRes, \
    obsXRes, obsTRes, y1, y2, x1, occluderFunc):

    startTime = 2*y2
    endTime = 2*sqrt(x1**2 + (y1+y2)**2)

    totalTime = endTime - startTime
    timeBinSize = totalTime / obsTRes

    returnMat = []

    r = 1

#    for j, sceneY in enumerate(np.linspace(0, y1, sceneYRes)):
#        for i, sceneX in enumerate(np.linspace(0, x1, sceneXRes)):
    for i, sceneX in enumerate(np.linspace(0, x1, sceneXRes)):
        for j, sceneY in enumerate(np.linspace(0, y1, sceneYRes)):

            sceneCoord = np.array([sceneX, sceneY])
            illuminationCoord = np.array([illuminationX, y1+y2])

            illuminationDistance = np.linalg.norm(sceneCoord - illuminationCoord)

            matColumn = []

            for k, obsX in enumerate(np.linspace(0, x1, obsXRes)):

                measurementCoord = np.array([obsX, y1+y2])
                measurementDistance = np.linalg.norm(sceneCoord - measurementCoord)
                totalTime = measurementDistance + illuminationDistance

                occlusionCoord = obsX*(y2*(1-r) + y1-sceneY) + sceneX*(y2*r)

                for t, obsT in enumerate(np.linspace(startTime+timeBinSize/2, \
                    endTime-timeBinSize/2, obsTRes)):

#                    print obsT, illuminationDistance, measurementDistance, timeBinSize / 2

                    if (abs(obsT-totalTime) < timeBinSize / 2) and \
                        occluderFunc(occlusionCoord, x1):

                        matColumn.append(1)
                    else:
                        matColumn.append(0)

#            print matColumn


            returnMat.append(matColumn)

    return np.transpose(np.array(returnMat))

def buildActiveTransferMatrixWithCorner(sceneXRes, sceneYRes, \
    obsThetaRes, obsTRes, x, y, r):

    diameterOfScene = sqrt(x*x + y*y)

    startTime = r
    endTime = 2*diameterOfScene + r

    totalTime = endTime - startTime
    timeBinSize = totalTime / obsTRes

    returnMat = []

    for i, sceneX in enumerate(np.linspace(0, x, sceneXRes)):
        for j, sceneY in enumerate(np.linspace(0, y, sceneYRes)):

            sceneCoord = np.array([sceneX, sceneY])
            illuminationCoord = np.array([0, 0])

            illuminationDistance = np.linalg.norm(sceneCoord - illuminationCoord)

            matColumn = []

            for k, obsTheta in enumerate(np.linspace(0, pi/2, obsThetaRes)):

                measurementCoord = np.array([-r*cos(obsTheta), -r*sin(obsTheta)])

                measurementDistance = np.linalg.norm(sceneCoord - measurementCoord)
                totalTime = measurementDistance + illuminationDistance

                x0 = sceneX
                y0 = sceneY
                x1 = -r*cos(obsTheta)
                y1 = -r*sin(obsTheta)

                cornerYIntercept = (y0 - y1 * x0/x1)/(1 - x0/x1)

                for t, obsT in enumerate(np.linspace(startTime+timeBinSize/2, \
                    endTime-timeBinSize/2, obsTRes)):

#                    print obsT, illuminationDistance, measurementDistance, timeBinSize / 2

                    if (abs(obsT-totalTime) < timeBinSize / 2) and \
                        (cornerYIntercept >= 0):

                        k = 1

                        attenuationFactor = 1/(measurementDistance**2)
#                        attenuationFactor = 1

                        matColumn.append(attenuationFactor)
                    else:
                        matColumn.append(0)

            returnMat.append(matColumn)

    return np.transpose(np.array(returnMat))
#    return np.array(returnMat)

def buildActiveTransferMatrixWithCornerNew(sceneXRes, sceneYRes, \
    obsThetaRes, obsTRes, x, y, r):

    diameterOfScene = sqrt(x*x + y*y)

    startTime = r
    endTime = 2*diameterOfScene + r

    totalTime = endTime - startTime
    timeBinSize = totalTime / obsTRes

    returnMat = []

    for i, sceneX in enumerate(np.linspace(0, x, sceneXRes)):
        for j, sceneY in enumerate(np.linspace(0, y, sceneYRes)):

            sceneCoord = np.array([sceneX, sceneY])
            illuminationCoord = np.array([0, 0])

            illuminationDistance = np.linalg.norm(sceneCoord - illuminationCoord)

            matColumn = []
            
            for t, obsT in enumerate(np.linspace(startTime+timeBinSize/2, \
                endTime-timeBinSize/2, obsTRes)):

                for k, obsTheta in enumerate(np.linspace(0, pi/2, obsThetaRes)):

                    measurementCoord = np.array([-r*cos(obsTheta), -r*sin(obsTheta)])

                    measurementDistance = np.linalg.norm(sceneCoord - measurementCoord)
                    totalTime = measurementDistance + illuminationDistance

                    x0 = sceneX
                    y0 = sceneY
                    x1 = -r*cos(obsTheta)
                    y1 = -r*sin(obsTheta)

                    cornerYIntercept = (y0 - y1 * x0/x1)/(1 - x0/x1)

#                    print obsT, illuminationDistance, measurementDistance, timeBinSize / 2

                    if (abs(obsT-totalTime) < timeBinSize / 2) and \
                        (cornerYIntercept >= 0):

                        k = 1

                        attenuationFactor = 1/(measurementDistance**2)
#                        attenuationFactor = 1

                        matColumn.append(attenuationFactor)
                    else:
                        matColumn.append(0)

            returnMat.append(matColumn)

    return np.transpose(np.array(returnMat))
#    return np.array(returnMat)

def randomOccluderFuncMaker(numTransitions):
    transitions = sorted([random.random() for _ in range(numTransitions)])

    def randomOccluderFunc(occlusionCoord, x1):
        loc = occlusionCoord/x1

        returnVal = True

        for i in transitions:
            if i < loc:
                returnVal = not returnVal

            else:
                return returnVal

        return returnVal

    return randomOccluderFunc

def displayOccluderFunc(occluderFunc, listSize = 200):

#    print occluderFunc(0, listSize)

    binaryList = [1*occluderFunc(i, listSize) for i in range(listSize)]
    rgbBinList = np.array([[i*255, i*255, i*255] for i in binaryList])

    viewFlatFrame(rgbBinList, 200)

def make2DTransferMatrix(sceneDimensions, occluder):
    sceneMaxX = sceneDimensions[0]
    sceneMaxY = sceneDimensions[1]

    occluderMaxX = occluder.shape[0]
    occluderMaxY = occluder.shape[1]

    assert occluderMaxX == sceneMaxX*2-1
    assert occluderMaxY == sceneMaxY*2-1

    returnMat = []

    for minX in range(sceneMaxX):
        for minY in range(sceneMaxY):
            matRow = []

            for x in range(sceneMaxX):
                for y in range(sceneMaxY):
                    matRow.append(occluder[minX+x][minY+y])

            returnMat.append(matRow)

    return np.array(returnMat)

def getPseudoInverse(transferMat, imSpatialDimensions, snr, beta):
    d = transferMat.shape[0]

    attenuationMat = getAttenuationMatrix(imSpatialDimensions, beta)
    doubleDft = np.kron(dft(imSpatialDimensions[1])/sqrt(imSpatialDimensions[1]), \
        dft(imSpatialDimensions[0])/sqrt(imSpatialDimensions[0]))
    diagArray = attenuationMat.flatten()
    priorMat = np.dot(np.dot(doubleDft, np.diag(diagArray)), np.conj(np.transpose(doubleDft)))

#    print transferMat.shape
#    print priorMat.shape
#    print np.dot(np.dot(transferMat, priorMat), np.transpose(transferMat)).shape

    miMat = snr * np.dot(np.dot(transferMat, priorMat), np.transpose(transferMat)) + np.identity(d)

    return np.dot(np.transpose(transferMat), np.linalg.inv(miMat))

def makeEllipseSnapshot(sceneDimensions, focus1, focus2, semiMajor, verbose=False):
    sceneMaxX = sceneDimensions[0]
    sceneMaxY = sceneDimensions[1]

#    print sceneDimensions, focus1, focus2, semiMajor

    mat = []

    for x in range(sceneMaxX):
        mat.append([])

        for y in range(sceneMaxY):

            if sqrt((x - focus1[0])**2 + (y - focus1[1])**2) + \
                sqrt((x - focus2[0])**2 + (y - focus2[1])**2) > 2*semiMajor:

                mat[-1].append(1)

            else:
#                print "hi"
                mat[-1].append(0)


#    viewFrame()

#    if (random.random() < 0.005):
    
    if verbose:
        viewFrame(imageify(np.array(mat)))
        distanceAttenuationMat = distanceAttenuation(sceneDimensions, min(sceneMaxX, sceneMaxY)*3, \
            np.array([sceneMaxX - focus1[0], sceneMaxY - focus1[1]]), flat=False)

        viewFrame(imageify(np.multiply(np.array(mat), distanceAttenuationMat)), adaptiveScaling=True)

    return np.array(mat).flatten()

def distanceAttenuation(sceneDimensions, distance, centerPoint, flat=True):

    sceneMaxX = sceneDimensions[0]
    sceneMaxY = sceneDimensions[1]

#    print sceneDimensions, focus1, focus2, semiMajor

    centerX = centerPoint[0]
    centerY = centerPoint[1]

    mat = []

    for x in range(sceneMaxX):
        mat.append([])

        for y in range(sceneMaxY):

            mat[-1].append(distance/((centerX-x)**2 + (centerY-y)**2 + distance**2)**(3/2))

#    viewFrame()

#    if (random.random() < 0.005):
    
#    if verbose:
#        viewFrame(imageify(np.array(mat)))

    if flat:
        return np.array(mat).flatten()
    else:
        return np.array(mat)

def makeSphereTransferMatrix(sceneDimensions):
    distanceToSphere = 30
    sphereRadius = 1/4

    planeDistance = 1

    sceneMaxX = sceneDimensions[0]
    sceneMaxY = sceneDimensions[1]

    centerX = sceneDimensions[0]/2
    centerY = sceneDimensions[1]/2

    center = np.array([centerX, centerY])

    sphereMat = []
    diskMat = []
    nearMat = []

    for x in range(sceneMaxX):
        for y in range(sceneMaxY):

            semiMinor = sphereRadius*min(sceneDimensions[0], sceneDimensions[1])
            distanceFromCenter = sqrt((x - centerX)**2 + (y - centerY)**2)
            eccentricity = distanceFromCenter / sqrt(distanceToSphere**2 + distanceFromCenter**2)
            semiMajor = semiMinor/sqrt(1 - eccentricity**2)
            focalDistance = sqrt(semiMajor**2 - semiMinor**2)
            ellipseCenter = np.array([x, y])
            towardsEllipseVector = ellipseCenter - center
            towardsEllipseUnitVector = towardsEllipseVector / np.linalg.norm(towardsEllipseVector)
            focus1 = ellipseCenter + towardsEllipseUnitVector*focalDistance
            focus2 = ellipseCenter - towardsEllipseUnitVector*focalDistance
            focus1 = ellipseCenter + 1.5*towardsEllipseUnitVector*focalDistance
            focus2 = ellipseCenter - 0.5*towardsEllipseUnitVector*focalDistance

            if x == 10 and y == 10:
#            if False:

                disk = makeEllipseSnapshot(sceneDimensions, ellipseCenter, ellipseCenter, \
                    semiMinor, verbose=True)

                sphereMat.append(makeEllipseSnapshot(sceneDimensions, focus1, focus2, semiMajor, \
                    verbose=True))
                distanceAttenuationVec = distanceAttenuation(sceneDimensions, min(sceneMaxX, sceneMaxY)*3, \
                    np.array([sceneMaxX - x, sceneMaxY - y]))

                diskMat.append(disk)
                nearMat.append(np.multiply(disk, distanceAttenuationVec))
            else:
                disk = makeEllipseSnapshot(sceneDimensions, ellipseCenter, ellipseCenter, \
                    semiMinor, verbose=False)
                sphereMat.append(makeEllipseSnapshot(sceneDimensions, focus1, focus2, semiMajor, \
                    verbose=False))
                distanceAttenuationVec = distanceAttenuation(sceneDimensions, min(sceneMaxX, sceneMaxY)*3, \
                    np.array([sceneMaxX - x, sceneMaxY - y]))                
                diskMat.append(makeEllipseSnapshot(sceneDimensions, ellipseCenter, ellipseCenter, \
                    semiMinor, verbose=False))
                nearMat.append(np.multiply(disk, distanceAttenuationVec))

    return np.array(sphereMat), np.array(diskMat), np.array(nearMat)

def estimateBinaryOccluderFromSunsShadow(obs, highVal):
    intensityArray = np.mean(obs, axis=2)

    obsShape = obs.shape

    print "obsShape", obsShape

    averageIntensity = np.mean(intensityArray)

    def greaterThanAverage(x):
        return 1*(x>averageIntensity)

    estimatedSmallOccluder = highVal*np.vectorize(greaterThanAverage)(intensityArray)

    viewSingleOccluder(estimatedSmallOccluder)

    fullOccluder = np.pad(estimatedSmallOccluder, [(int(obsShape[0]/2), int(obsShape[0]/2-1)), \
        (int(obsShape[1]/2), int(obsShape[1]/2-1))], "constant", constant_values=highVal)

    return fullOccluder

def verticalTransferMatrix(sceneDimensions):
    sceneMaxX = sceneDimensions[0]
    sceneMaxY = sceneDimensions[1]

    returnMat = []

    for sceneX in range(sceneMaxX):
        matRow = []

        for x in range(sceneMaxX):
            for y in range(sceneMaxY):
                if x == sceneX:
                    matRow.append(1)
                else:
                    matRow.append(0)

        returnMat.append(matRow)

#    p.matshow(np.array(returnMat))
#    p.show()

    return np.array(returnMat)    

def horizontalTransferMatrix(sceneDimensions):
    sceneMaxX = sceneDimensions[0]
    sceneMaxY = sceneDimensions[1]

    returnMat = []

    for sceneY in range(sceneMaxY):
        matRow = []

        for x in range(sceneMaxX):
            for y in range(sceneMaxY):
                if y == sceneY:
                    matRow.append(1)
                else:
                    matRow.append(0)

        returnMat.append(matRow)

#    p.matshow(np.array(returnMat))
 #   p.show()

    return np.array(returnMat)     

if __name__ == "__main__":

    if FOURIER_UNDERSTANDER:
        n = 10

        x = np.random.random(n)

        d = dft(n)

        print np.dot(d, x)
        print np.fft.fft(x)

    if FOURIER_UNDERSTANDER_2D:
        n = 5

        x = np.random.random((n, n))

    #    d = dft(n)

    #    print np.dot(d, x)
        print np.fft.fft2(x)

    if MATRIX_SIZE_TEST:
        n = 7000
        A = np.random.random((n, n))

        p.matshow(A)
        p.show()

        Ainv = np.linalg.inv(A)
        p.matshow(Ainv)
        p.show()

    if ALRIGHT_LETS_DO_THIS:
    #    imRaw = Image.open("winnie.png")
    #    im = np.array(imRaw).astype(float)

        IM_BRIGHTNESS = 1e10
        THERMAL_NOISE = 1e12
    #    THERMAL_NOISE = 1


        beta = 0.1
    #    beta = 1

    #    snr = IM_BRIGHTNESS / THERMAL_NOISE

        snr = 3e6
    #    snr = 1e12

    #    im = pickle.load(open("winnie_downsampled_a_lot.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("calvin_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_clipped.p", "r"))*IM_BRIGHTNESS
        im = pickle.load(open("dora_slightly_downsampled.p", "r"))*IM_BRIGHTNESS

        imSpatialDimensions = im.shape[:2]

        numPixels = imSpatialDimensions[0]*imSpatialDimensions[1]

        power = np.sum(np.sum(np.sum(np.multiply(im, im), 0), 0), 0)/(numPixels*3)

        print "power", power

        #print im.shape



        print imSpatialDimensions

    #    im = makeImpulseImage(imSpatialDimensions, "center")*IM_BRIGHTNESS
    #    im = makeSquareImage(imSpatialDimensions, 1/4)*IM_BRIGHTNESS

        viewFrame(im, 1e-10)
    #    viewFrame(im, 1e-10, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)
    #    viewFrame(np.flip(np.flip(im, 0), 1), 1e-10)
    #    viewFrame(averageVertically(im), 1e-10)


    #    occluderWindow = pinCircle(imSpatialDimensions, 1/8)
        occluderWindow = circleSpeck((28, 28), 1/5)
#        occluderWindow = circleSpeck(imSpatialDimensions, 1/4)
    #    occluderWindow = coarseCheckerBoard(imSpatialDimensions)
    #    occluderWindow = fineCheckerBoard(imSpatialDimensions)
    #    occluderWindow = pinSquare(imSpatialDimensions, 1/4)
    #    occluderWindow = squareSpeck(imSpatialDimensions, 1/4)
    #    occluderWindow = pinHole(imSpatialDimensions)
    #    occluderWindow = pinSpeck(imSpatialDimensions)
    #    occluderWindow = randomOccluder(imSpatialDimensions)
    #    occluderWindow = lens2(imSpatialDimensions)
    #    occluderWindow = verticalColumn(imSpatialDimensions, 1/4)
    #    occluderWindow = spectrallyFlatOccluder(imSpatialDimensions)

    #    viewRepeatedOccluder(occluderWindow)
        viewSingleOccluder(occluderWindow)

        pickle.dump(occluderWindow, open("circle_28_28.p", "w"))

        bits = getBitsOfOccluder(occluderWindow, beta, snr*100)
        pixels = bits/log(IM_BRIGHTNESS/THERMAL_NOISE*100 + 1, 2)
        sideLength = sqrt(pixels)

        R = bits / (imSpatialDimensions[0]*imSpatialDimensions[1])

        D = power*exp(-2*R)

        print "D:", D

        print bits, pixels, sideLength

        obsPlaneIm = doFuncToEachChannel(convolve2dMaker(occluderWindow), im)

    #    viewFrame(obsPlaneIm, 3e-10)

        noisyObsPlaneIm = addNoise(obsPlaneIm, THERMAL_NOISE)

        viewFrame(noisyObsPlaneIm, 1e-10)

    #    recoveryWindowSimple = getRecoveryWindowSimple(occluderWindow)
    #    recoveredImSimple = doFuncToEachChannel(convolve2dMaker(recoveryWindowSimple), \
    #        noisyObsPlaneIm)

    #    recoveryWindowSophisticated = getRecoveryWindowSophisticated(occluderWindow, beta, snr)
        recoveryWindowSophisticated = getRecoveryWindowOneOverF(occluderWindow, snr)

        recoveredImSophisticated = doFuncToEachChannel(convolve2dMaker(recoveryWindowSophisticated), \
            noisyObsPlaneIm)


    #    print THERMAL_NOISE * np.ones(im.shape)

    #    print recoveredIm

    #    recoveredIm -= THERMAL_NOISE * np.ones(im.shape)

    #    print recoveredIm

    #    viewFrame(obsPlaneIm, 2e-10)
    #    viewFrame(recoveredImSimple, 5e-11)
        print "recovered image: lower right"
    #    viewFrame(recoveredImSophisticated, 9e-11)

    #    averageObsPlaneIntensity = sum(sum(noisyObsPlaneIm, 0), 0)/numPixels
    #    averageObsPlane = np.kron(np.ones(imSpatialDimensions), averageObsPlaneIntensity)

    #    print averageObsPlane.shape


    #    print recoveredImSophisticated, 3*noisyObsPlaneIm

    #    print recoveredImSophisticated - 3*noisyObsPlaneIm

        pickle.dump(recoveredImSophisticated, open("recovered_im_2.p", "w"))

        viewFrame(recoveredImSophisticated, adaptiveScaling=True)
    #    viewFrame(recoveredImSophisticated, 1e-10, differenceImage=True, meanSubtraction=True)
    #    viewFrame(recoveredImSophisticated, 1e-9, differenceImage=True, meanSubtraction=True)

    #    viewFrame(recoveredImSophisticated, 1e-10, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)
    #    viewFrame(recoveredImSophisticated, 1e-9, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)
    #    viewFrame(recoveredImSophisticated, 1e-8, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)


    #    viewFrame(recoveredImSophisticated - 3*noisyObsPlaneIm, 1e-10, differenceImage=True)
    #    viewFrame(recoveredImSophisticated - 3*noisyObsPlaneIm, 1e-8, differenceImage=True)

    #    viewFrame(recoveredImSophisticated, 6e-11)
    #    viewFrame(recoveredImSophisticated, 2e-11)

    #    viewFrame(recoveredImSophisticated, 3e-9)


        diff = im - recoveredImSophisticated
    #    diff = averageVertically(im) - recoveredImSophisticated
    #    print diff

        print im
        print recoveredImSophisticated

        empD = np.real(np.sum(np.sum(np.sum(np.multiply(diff, diff), 0), 0), 0)/(numPixels*3))

        print "empirical D:", empD*1e-20

        print "reconstruction SNR:", 10*log(power/empD, 10)

    #    viewFrame(recoveredImSophisticated, 3e-9)

    if TOEPLITZ_2D:

        IM_BRIGHTNESS = 1e10
    #    THERMAL_NOISE = 1e8
        THERMAL_NOISE = 1

        beta = 0.3
    #    beta = 1

        snr = 3e6
    #    snr = 1e12

    #    im = pickle.load(open("winnie_downsampled_a_lot.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("calvin_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_clipped.p", "r"))*IM_BRIGHTNESS
        im = pickle.load(open("dora_slightly_downsampled.p", "r"))*IM_BRIGHTNESS

        imSpatialDimensions = im.shape[:2]

        numPixels = imSpatialDimensions[0]*imSpatialDimensions[1]

        power = np.sum(np.sum(np.sum(np.multiply(im, im), 0), 0), 0)/(numPixels*3)

        print "power", power

        #print im.shape



        for i in range(1,101):

            print i, "/", 100


            occluderWindow = pinCircle((2*imSpatialDimensions[0]-1,
                2*imSpatialDimensions[1]-1), i/256)

            inversionOccluderWindow = cornerCircle(imSpatialDimensions, i/128)

            viewRepeatedOccluder(inversionOccluderWindow,
                filename="inversionocc" + padIntegerWithZeros(i, 2) + ".png")
            viewSingleOccluder(occluderWindow,
                filename="trueocc" + padIntegerWithZeros(i, 2) + ".png")

            bits = getBitsOfOccluder(occluderWindow, beta, snr*100)
            pixels = bits/log(IM_BRIGHTNESS/THERMAL_NOISE*100 + 1, 2)
            sideLength = sqrt(pixels)

            R = bits / (imSpatialDimensions[0]*imSpatialDimensions[1])

            D = power*exp(-2*R)

            obsPlaneIm = doFuncToEachChannel(convolve2dMakerToeplitz(occluderWindow), im)

            noisyObsPlaneIm = addNoise(obsPlaneIm, THERMAL_NOISE)

            recoveryWindowSophisticated = getRecoveryWindowSophisticated(inversionOccluderWindow, beta, snr)
            recoveredImSophisticated = doFuncToEachChannel(convolve2dMaker(recoveryWindowSophisticated), \
                noisyObsPlaneIm)


            print "recovered image: lower right"


            viewFrame(recoveredImSophisticated, 2e-10, filename="recovery" + padIntegerWithZeros(i, 2) + ".png")

            diff = im - recoveredImSophisticated
        #    diff = averageVertically(im) - recoveredImSophisticated
        #    print diff

            print im
            print recoveredImSophisticated

            empD = np.real(np.sum(np.sum(np.sum(np.multiply(diff, diff), 0), 0), 0)/(numPixels*3))

            print "empirical D:", empD*1e-20

            print "reconstruction SNR:", 10*log(power/empD, 10)

    #    viewFrame(recoveredImSophisticated, 3e-9)

    if ANTONIO_METHOD:

        IM_BRIGHTNESS = 1e10
    #    THERMAL_NOISE = 1e8
        THERMAL_NOISE = 1

        beta = 0.3
    #    beta = 1

        snr = 3e6
    #    snr = 1e12

    #    im = pickle.load(open("winnie_downsampled_a_lot.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("calvin_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_clipped.p", "r"))*IM_BRIGHTNESS
        im = pickle.load(open("dora_slightly_downsampled.p", "r"))*IM_BRIGHTNESS

        imSpatialDimensions = im.shape[:2]

        numPixels = imSpatialDimensions[0]*imSpatialDimensions[1]

        power = np.sum(np.sum(np.sum(np.multiply(im, im), 0), 0), 0)/(numPixels*3)

        print "power", power

        #print im.shape




        occluderWindow = coarseCheckerBoard((2*imSpatialDimensions[0],
            2*imSpatialDimensions[1]))

        obsPlaneIm = doFuncToEachChannel(convolve2dMakerToeplitz(occluderWindow), im)

        noisyObsPlaneIm = addNoise(obsPlaneIm, THERMAL_NOISE)

        doubleObsPlane = np.concatenate((noisyObsPlaneIm, noisyObsPlaneIm), 0)
        quadrupleObsPlane = np.concatenate((doubleObsPlane, doubleObsPlane), 1)

        recoveryWindowSophisticated = getRecoveryWindowSophisticated(occluderWindow, beta, snr)
        recoveredImSophisticated = doFuncToEachChannel(convolve2dMaker(recoveryWindowSophisticated), \
            quadrupleObsPlane)


        print "recovered image: lower right"


        viewFrame(recoveredImSophisticated, 5e-11)

        diff = im - recoveredImSophisticated
    #    diff = averageVertically(im) - recoveredImSophisticated
    #    print diff

        print im
        print recoveredImSophisticated

        empD = np.real(np.sum(np.sum(np.sum(np.multiply(diff, diff), 0), 0), 0)/(numPixels*3))

        print "empirical D:", empD*1e-20

        print "reconstruction SNR:", 10*log(power/empD, 10)

    #    viewFrame(recoveredImSophisticated, 3e-9)


    if TOEPLITZ:
        n = 104

        dftMat = dft(n)

        A = upperTriangular(n)
    #    A = circulant(np.random.randint(0, 2, n))
    #    A = circulant([1]*int(n/2)+[0]*int(n/2))

    #    A = toeplitz([1] + [1*(random.random()<0.5) for _ in range(n-1)], \
    #        [1] + [1*(random.random()<0.5) for _ in range(n-1)])

    #    A = toeplitz([1]*int(n/8) + [0]*int(7*n/8), [1]*int(n/8) + [0]*int(7*n/8))

    #    A = circulant([1]*int(n/8) + [0]*int(6*n/8) + [1]*int(n/8))

        p.matshow(A, cmap="Greys_r")
        p.show()

        res = np.dot(np.dot(dftMat, A), np.conj(np.transpose(dftMat)))
        res[0][0] = 0
    #    res[1][1] = 0
    #    res[n-1][n-1] = 0
    #    res[2][2] = 0
    #    res[n-2][n-2] = 0
    #    res[3][3] = 0
    #    res[n-3][n-3] = 0

        p.matshow(np.real(res))
        p.colorbar()
        p.show()

    if DIFFERENT_DEPTHS_SIM:

        IM_BRIGHTNESS = 1e10
        THERMAL_NOISE = 1e10
    #    THERMAL_NOISE = 1

        beta = 0.1
    #    beta = 1

        snr = 1e2
    #    snr = 1e12

    #    im = pickle.load(open("winnie_downsampled_a_lot.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("calvin_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_clipped.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("dora_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
        im = pickle.load(open("dora_very_downsampled.p", "r"))*IM_BRIGHTNESS


        imShape = im.shape
        imSpatialDimensions = imShape[:2]

        viewFrame(im, 1e-10)
    #    vec = imageToVector(im)
    #    viewFlatFrame(vec, 200, magnification=1e-10)
    #    im = vectorToImage(vec, imSpatialDimensions)
    #    viewFrame(im, 1e-10)

        imVector = imageToVector(im)

        imHeight = imSpatialDimensions[0]
        imWidth = imSpatialDimensions[1]

    #    occluderWindow = oneDPinHole(2*imSpatialDimensions[1]-1, 1/4)
    #    occluderWindow = oneDRandom(2*imSpatialDimensions[1]-1)
    #    occluderWindow = oneDNothing(2*imSpatialDimensions[1]-1)
        occluderWindow = oneDedge(2*imSpatialDimensions[1]-1)

        d1 = 1
        d2 = 1
        d3 = 1

        subMatrices = [getToeplitzLikeTransferMatrixWithVariedDepth(occluderWindow, \
            (d2+(i/imHeight)*d1)/(d3+d2+(i/imHeight)*d1), d3/((d3+d2+(i/imHeight)*d1))) for i in range(imHeight)]

        transferMatrix = np.concatenate(subMatrices, axis=1)

        p.matshow(transferMatrix, cmap="Greys_r")
        p.show()

        obsPlane = doFuncToEachChannelVec(lambda x: np.dot(transferMatrix, x), imVector)

        viewFlatFrame(obsPlane, 10, magnification=1e-11)

        n = imVector.shape[0]

        attenuationMat = getAttenuationMatrix(imSpatialDimensions, beta)

        p.matshow(attenuationMat)
        p.colorbar()
        p.show()

        print "computing prior mat"

    #    doubleDft = np.kron(dft(imShape[0])/sqrt(imShape[0]), dft(imShape[1])/sqrt(imShape[1]))
        attenuationMat = getAttenuationMatrix(imSpatialDimensions, beta)
        doubleDft = np.kron(dft(imShape[1])/sqrt(imShape[1]), dft(imShape[0])/sqrt(imShape[0]))
        diagArray = attenuationMat.flatten()
        priorMat = np.dot(np.dot(doubleDft, np.diag(diagArray)), np.conj(np.transpose(doubleDft)))


        print priorMat.shape

        p.matshow(np.real(priorMat))
        p.colorbar()
        p.show()


        print "computing pseudo inverse"

        miMat = snr*np.dot(np.dot(transferMatrix, priorMat), np.transpose(transferMatrix)) + np.identity(imWidth)
        mi = np.linalg.slogdet(miMat)[1]

        print "mi", mi, mi/imWidth

        pseudoInverse = snr*np.linalg.inv(miMat)
        recoveryMat = np.dot(np.transpose(transferMatrix), pseudoInverse)

        recoveredScene = doFuncToEachChannelVec(lambda x: np.dot(recoveryMat, x), obsPlane)

        viewFlatFrame(recoveredScene, 200, magnification=1e-10)

        im = vectorToImage(recoveredScene, imSpatialDimensions)

        viewFrame(im, magnification=6e-11)

    if ACTIVE_SIM:

        im = pickle.load(open("shapes_very_downsamples.p", "r"))

        y1 = 1
        y2 = 1
        x1 = 1

        x = 1
        y = 1
        r = 1

    #    viewFrame(im)

        imSpatialDimensions = im.shape[:-1]

        sceneXRes = im.shape[0]
        sceneYRes = im.shape[1]
        obsXRes = im.shape[0]
#        obsThetaRes = obsXRes
#        obsTRes = 40
        obsThetaRes = 16
        obsTRes = 63

        beta = 0.1
        snr = 1e0

    #    imVector = imageToVector(np.flip(np.flip(np.swapaxes(im, 0, 1), 0), 1))
        imVector = imageToVector(np.swapaxes(im, 0, 1))
    #    imVector = imageToVector(im)

        illuminationX = x1/2

        attenuationMat = getAttenuationMatrix(imSpatialDimensions, beta)

    #    transferMatrix = buildActiveTransferMatrixWithEdge(illuminationX, sceneXRes, sceneYRes, \
    #        obsXRes, obsTRes, y1, y2, x1)

    #    transferMatrix = buildActiveTransferMatrixNoCorner(illuminationX, sceneXRes, sceneYRes, \
    #        obsXRes, obsTRes, y1, y2, x1)

        transferMatrix = buildActiveTransferMatrixWithCorner(sceneXRes, sceneYRes, \
            obsThetaRes, obsTRes, x, y, r)

#        numTransitions = 25

#        randomOccluderFunc = randomOccluderFuncMaker(numTransitions)

#        transferMatrix = buildActiveTransferMatrixWithOccluder(illuminationX, sceneXRes, sceneYRes, \
#            obsXRes, obsTRes, y1, y2, x1, randomOccluderFunc)



    #    p.matshow(transferMatrix)
    #    p.show()

    #    p.matshow(transferMatrix)
    #    p.colorbar()
    #    p.show()

#        displayOccluderFunc(randomOccluderFunc)


        obsPlane = doFuncToEachChannelVec(lambda x: np.dot(transferMatrix, x), imVector)
        print obsPlane.shape
        p.plot(np.transpose(obsPlane)[0])
        p.show()


    #    obsPlane = addExpNoise(obsPlane, 1e-10)

        doubleDft = np.kron(dft(sceneYRes)/sqrt(sceneYRes), dft(sceneXRes)/sqrt(sceneXRes))
        diagArray = attenuationMat.flatten()
        priorMat = np.dot(np.dot(doubleDft, np.diag(diagArray)), np.conj(np.transpose(doubleDft)))

        miMat = snr*np.dot(np.dot(transferMatrix, priorMat), np.transpose(transferMatrix)) + np.identity(obsXRes * obsTRes)

        pseudoInverse = snr*np.linalg.inv(miMat)
        recoveryMat = np.dot(np.transpose(transferMatrix), pseudoInverse)

        recoveredScene = doFuncToEachChannelVec(lambda x: np.dot(recoveryMat, x), obsPlane)

    #    viewFlatFrame(recoveredScene, 200, magnification=1)

        im = vectorToImage(recoveredScene, imSpatialDimensions)

    #    viewFrame(np.flip(np.flip(np.swapaxes(im, 0, 1), 0), 1), magnification=1)
        viewFrame(np.swapaxes(im, 0, 1), magnification=1)
    #    viewFrame(im, magnification=1)


    #    print np.sum(np.sum(im))

        mi = np.linalg.slogdet(miMat)[1]

        print "mi", mi/(sceneXRes*sceneYRes)

    #    print obsPlane



    #    transferMatrix =

    #    getToeplitzLikeTransferMatrixWithVariedDepth(occluder, d1, d2)

    if ANGLE_CORNER:

    #    im = pickle.load(open("dora_very_downsampled.p", "r"))
        im = pickle.load(open("impulse_5_6_11_11.p", "r"))

        viewFrame(im)

        beta = 0.1
        snr = 1e4

        imShape = im.shape[:-1]
        imSpatialDimensions = imShape
        imVector = imageToVector(np.swapaxes(im, 0, 1))

        attenuationMat = getAttenuationMatrix(imSpatialDimensions, beta)

        vector1 = np.array([sqrt(2)/2,0,sqrt(2)/2])
        vector2 = np.array([0,1,0])
        vertRadius = 1
        horizRadius = 1

        transferMatrix = makeCornerTransferMatrix(vector1, vector2, imShape, vertRadius, horizRadius)

        obsPlane = doFuncToEachChannelVec(lambda x: np.dot(transferMatrix, x), imVector)

        viewFrame(vectorToImage(obsPlane, imShape), 1e2)

        doubleDft = np.kron(dft(sceneYRes)/sqrt(sceneYRes), dft(sceneXRes)/sqrt(sceneXRes))
        diagArray = attenuationMat.flatten()
        priorMat = np.dot(np.dot(doubleDft, np.diag(diagArray)), np.conj(np.transpose(doubleDft)))

        miMat = snr*np.dot(np.dot(transferMatrix, priorMat), np.transpose(transferMatrix)) + np.identity(obsXRes * obsTRes)

        pseudoInverse = snr*np.linalg.inv(miMat)
        recoveryMat = np.dot(np.transpose(transferMatrix), pseudoInverse)

        recoveredScene = doFuncToEachChannelVec(lambda x: np.dot(recoveryMat, x), obsPlane)

    if MAKE_ELEPHANT:
        viewFrame(imageify(elephant((100, 100), 1/2)), adaptiveScaling=True)

    if LIGHT_OF_THE_SUN_SIM:

        SUN_BRIGHTNESS = 5e11
    #    SUN_BRIGHTNESS = 1e10
        IM_BRIGHTNESS = 1e10
        THERMAL_NOISE = 1e-3

        SUN_RADIUS = 1
    #    THERMAL_NOISE = 1

        usePrior = False

        beta = 0.3

        snr = 1e15

        im = pickle.load(open("dora_very_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("dora_extremely_downsampled.p", "r"))*IM_BRIGHTNESS

        imSpatialDimensions = im.shape[:2]
        highVal = 1/(imSpatialDimensions[0]*imSpatialDimensions[1])

        numPixels = imSpatialDimensions[0]*imSpatialDimensions[1]

        power = np.sum(np.sum(np.sum(np.multiply(im, im), 0), 0), 0)/(numPixels*3)

        print "power", power

        print imSpatialDimensions

    #    occluderWindow = cornerSpeck(imSpatialDimensions, 1/8)
        occluderWindow = circleSpeck((imSpatialDimensions[0]*2-1, imSpatialDimensions[1]*2-1), 1/8)

        transferMatrix = make2DTransferMatrix(imSpatialDimensions, occluderWindow)

        sun = SUN_BRIGHTNESS*imageify(pinCircle(imSpatialDimensions, 1/15, highVal=1))

    #    viewFrame(sun, 1e-10)

        print np.mean(im)
        print np.mean(SUN_BRIGHTNESS*imageify(pinCircle(imSpatialDimensions, 1, highVal=1)))

        im += sun

        viewFrame(im, 1e-10)
        viewSingleOccluder(occluderWindow)

        print "s1", occluderWindow.shape

        bits = getBitsOfOccluder(occluderWindow, beta, snr*100)
        pixels = bits/log(IM_BRIGHTNESS/THERMAL_NOISE*100 + 1, 2)
        sideLength = sqrt(pixels)

    #    print bits, pixels, sideLength

    #    obsPlaneIm = doFuncToEachChannel(convolve2dMaker(occluderWindow), im)
        obsPlaneVec = doFuncToEachChannelVec(lambda x: np.dot(transferMatrix, x), imageToVector(im))

    #    viewFrame(vectorToImage(obsPlaneIm), 3e-12)

        noisyObsPlaneVec = addNoise(obsPlaneVec, THERMAL_NOISE)

        noisyObsPlaneIm = vectorToImage(noisyObsPlaneVec, imSpatialDimensions)

        viewFrame(noisyObsPlaneIm, 3e-10)


        estimatedOccluder = estimateBinaryOccluderFromSunsShadow(noisyObsPlaneIm, highVal)

        viewSingleOccluder(estimatedOccluder)

        print "s2", estimatedOccluder.shape

    #    recoveryWindowSophisticated = getRecoveryWindowSophisticated(occluderWindow, beta, snr)

        estimatedTransferMatrix = make2DTransferMatrix(imSpatialDimensions, estimatedOccluder)

        if usePrior:

            attenuationMat = getAttenuationMatrix(imSpatialDimensions, beta)

            sceneXRes = im.shape[0]
            sceneYRes = im.shape[1]

            doubleDft = np.kron(dft(sceneYRes)/sqrt(sceneYRes), dft(sceneXRes)/sqrt(sceneXRes))
            diagArray = attenuationMat.flatten()
            priorMat = np.dot(np.dot(doubleDft, np.diag(diagArray)), np.conj(np.transpose(doubleDft)))

            miMat = snr*np.dot(np.dot(transferMatrix, priorMat), np.transpose(transferMatrix)) + np.identity(sceneXRes * sceneYRes)

            pseudoInverse = snr*np.linalg.inv(miMat)
            recoveryMat = np.dot(np.transpose(transferMatrix), pseudoInverse)

            recoveredScene = doFuncToEachChannelVec(lambda x: np.dot(recoveryMat, x), noisyObsPlaneVec)

        else:
            recoveredScene = doFuncToEachChannelVec(lambda x: np.dot(np.linalg.inv(estimatedTransferMatrix), x), noisyObsPlaneVec)
    #    recoveredImSophisticated = doFuncToEachChannel(convolve2dMaker(recoveryWindowSophisticated), \
    #        noisyObsPlaneIm)

        viewFrame(vectorToImage(recoveredScene, imSpatialDimensions), 8e-14)

    if SPHERE_DISK_COMPARISON:

        im = pickle.load(open("dora_very_downsampled.p", "r"))
        imSpatialDimensions = np.array(im).shape[:2]
        
    #    viewFrame(im)

        sphereMat, diskMat, nearMat = makeSphereTransferMatrix(imSpatialDimensions)

        obsSphere = doFuncToEachChannelVec(lambda x: np.dot(sphereMat, x), imageToVector(im))
        obsNear = doFuncToEachChannelVec(lambda x: np.dot(nearMat, x), imageToVector(im))
        obsDisk = doFuncToEachChannelVec(lambda x: np.dot(diskMat, x), imageToVector(im))

        viewFrame(vectorToImage(obsSphere, imSpatialDimensions), adaptiveScaling=True)
        viewFrame(vectorToImage(obsNear, imSpatialDimensions), adaptiveScaling=True)
        viewFrame(vectorToImage(obsDisk, imSpatialDimensions), adaptiveScaling=True)


        inverseMatrixSphere = np.dot(np.linalg.inv(np.dot(np.transpose(diskMat), diskMat) + \
            2e3*np.identity(diskMat.shape[0])), np.transpose(diskMat))

        inverseMatrixNear = np.dot(np.linalg.inv(np.dot(np.transpose(diskMat), diskMat) + \
            2e1*np.identity(diskMat.shape[0])), np.transpose(diskMat))

    #    viewFrame(vectorToImage(obs, imSpatialDimensions), adaptiveScaling=True)

        recoverySphere = doFuncToEachChannelVec(lambda x: np.dot(inverseMatrixSphere, x), obsSphere)
        recoveryNear = doFuncToEachChannelVec(lambda x: np.dot(inverseMatrixSphere, x), obsNear)


        viewFrame(vectorToImage(recoverySphere, imSpatialDimensions), adaptiveScaling=True, \
            magnification=1)

        viewFrame(vectorToImage(recoveryNear, imSpatialDimensions), adaptiveScaling=True, \
            magnification=1)

    if DISPERSION_PLOT:
        d = 10
        hs = np.linspace(0, 40, 100)
        intensities = [h/(h**2 + d**2)**(3/2) for h in hs]

        p.plot(hs, intensities)
        ax = p.gca()
        ax.set_ylabel("Intensity")
        ax.set_xlabel("h")
        p.axvline(x=d, color="black")
        p.show()

    if LOAD_BU_SIM_DATA:
        matFileName = "data.mat"

        mat = loadmat(matFileName)

        simData = 0

        photonCounts = batchAndDifferentiate(mat["measurement"][0][simData], [(100, False), (1, False)])
        photonCountsDiff = batchAndDifferentiate(mat["measurement"][0][simData], [(100, False), (1, True)])


        print photonCounts.shape


        imSpatialDimensions = (20, 20)

        sceneXRes = imSpatialDimensions[0]
        sceneYRes = imSpatialDimensions[1]
        obsThetaRes = 20
        obsTRes = 42

        x = 5
        y = 5
        r = 0.1

        beta = 0.1
        snr = 1e2
        obsPlane = np.transpose(photonCounts).flatten()

        print photonCounts.shape
        print obsPlane.shape

        attenuationMat = getAttenuationMatrix(imSpatialDimensions, beta)

        transferMatrix = buildActiveTransferMatrixWithCorner(sceneXRes, sceneYRes, \
            obsThetaRes, obsTRes, x, y, r)

        doubleDft = np.kron(dft(sceneYRes)/sqrt(sceneYRes), dft(sceneXRes)/sqrt(sceneXRes))
        diagArray = attenuationMat.flatten()
        priorMat = np.dot(np.dot(doubleDft, np.diag(diagArray)), np.conj(np.transpose(doubleDft)))

        miMat = snr*np.dot(np.dot(transferMatrix, priorMat), np.transpose(transferMatrix)) + np.identity(obsThetaRes * obsTRes)

        pseudoInverse = snr*np.linalg.inv(miMat)
        recoveryMat = np.dot(np.transpose(transferMatrix), pseudoInverse)

#        recoveredScene = doFuncToEachChannelVec(lambda x: np.dot(recoveryMat, x), obsPlane)
        recoveredScene = np.dot(recoveryMat, obsPlane)


    #    viewFlatFrame(recoveredScene, 200, magnification=1)

        im = vectorToImage(imageify(recoveredScene), imSpatialDimensions)

        viewFrame(im, adaptiveScaling=True, differenceImage=False, magnification=2)

        for i in range(obsThetaRes):
            p.plot(np.transpose(photonCountsDiff)[i])
            p.ylim(-1,1)
            p.show()


    if LOAD_BU_REAL_DATA:
        matFileName = "act_corner_2019_01_28_acq3.mat"

        mat = loadmat(matFileName)

        print mat.keys()
        print mat["binRes"]

        imSpatialDimensions = (20, 20)

        sceneXRes = imSpatialDimensions[0]
        sceneYRes = imSpatialDimensions[1]
        obsThetaRes = 16
        obsTRes = 63

#        obsThetaRes = 8
#        obsTRes = 31

#        obsXRes = sceneXRes
#        obsYRes = sceneYRes

        x = 5
        y = 5
        r = 5

        beta = 0.1
        snr = 1e2

        photonCounts = batchAndDifferentiate(mat["histData"], [(100, False), (1, False)])
        photonCountsDiff = batchAndDifferentiate(mat["histData"], [(100, False), (1, True)])

        obsPlane = np.transpose(photonCounts).flatten()
#        obsPlane = photonCounts.flatten()

        p.plot(obsPlane)
        p.show()

        print photonCounts.shape
        print obsPlane.shape

        attenuationMat = getAttenuationMatrix(imSpatialDimensions, beta)

#        transferMatrix = buildActiveTransferMatrixWithCornerNew(sceneXRes, sceneYRes, \
#            obsThetaRes, obsTRes, x, y, r)
        transferMatrix = buildActiveTransferMatrixWithCorner(sceneXRes, sceneYRes, \
            obsThetaRes, obsTRes, x, y, r)

        p.matshow(transferMatrix)
        p.show()

        doubleDft = np.kron(dft(sceneYRes)/sqrt(sceneYRes), dft(sceneXRes)/sqrt(sceneXRes))
        diagArray = attenuationMat.flatten()
        priorMat = np.dot(np.dot(doubleDft, np.diag(diagArray)), np.conj(np.transpose(doubleDft)))

        miMat = snr*np.dot(np.dot(transferMatrix, priorMat), np.transpose(transferMatrix)) + np.identity(obsThetaRes * obsTRes)

        pseudoInverse = snr*np.linalg.inv(miMat)
        recoveryMat = np.dot(np.transpose(transferMatrix), pseudoInverse)

        p.matshow(np.real(recoveryMat))
        p.show()

#        recoveredScene = doFuncToEachChannelVec(lambda x: np.dot(recoveryMat, x), obsPlane)
        recoveredScene = np.dot(recoveryMat, obsPlane)

        p.plot(recoveredScene)
        p.show()

    #    viewFlatFrame(recoveredScene, 200, magnification=1)

        im = vectorToImage(imageify(recoveredScene), imSpatialDimensions)

        viewFrame(im, adaptiveScaling=True, differenceImage=False, magnification=2)

        for i in range(obsThetaRes):
            p.plot(np.transpose(photonCountsDiff)[i])
            p.ylim(-20,20)
            p.show()

    if JOIN_INFO:
    #    imRaw = Image.open("winnie.png")
    #    im = np.array(imRaw).astype(float)

        IM_BRIGHTNESS = 1e10
        THERMAL_NOISE = 1e10
    #    THERMAL_NOISE = 1


        beta = 0.1
    #    beta = 1

    #    snr = IM_BRIGHTNESS / THERMAL_NOISE

        snr = 1e4
    #    snr = 3e6
    #    snr = 1e12

    #    im = pickle.load(open("winnie_downsampled_a_lot.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("calvin_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_clipped.p", "r"))*IM_BRIGHTNESS
        im = pickle.load(open("dora_very_downsampled.p", "r"))*IM_BRIGHTNESS

        imSpatialDimensions = im.shape[:2]

        numPixels = imSpatialDimensions[0]*imSpatialDimensions[1]

        power = np.sum(np.sum(np.sum(np.multiply(im, im), 0), 0), 0)/(numPixels*3)

        print "power", power

        #print im.shape

        print imSpatialDimensions

    #    im = makeImpulseImage(imSpatialDimensions, "center")*IM_BRIGHTNESS
    #    im = makeSquareImage(imSpatialDimensions, 1/4)*IM_BRIGHTNESS

        viewFrame(im, 1e-10)
    #    viewFrame(im, 1e-10, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)
    #    viewFrame(np.flip(np.flip(im, 0), 1), 1e-10)
    #    viewFrame(averageVertically(im), 1e-10)


        occluderWindow1 = pinCircle([2*i-1 for i in imSpatialDimensions], 1/8)
    #    occluderWindow = circleSpeck(imSpatialDimensions, 1/4)
    #    occluderWindow = coarseCheckerBoard(imSpatialDimensions)
    #    occluderWindow = fineCheckerBoard(imSpatialDimensions)
    #    occluderWindow = pinSquare(imSpatialDimensions, 1/4)
    #    occluderWindow = squareSpeck(imSpatialDimensions, 1/4)
    #    occluderWindow = pinHole(imSpatialDimensions)
    #    occluderWindow = pinSpeck(imSpatialDimensions)
    #    occluderWindow = randomOccluder(imSpatialDimensions)
    #    occluderWindow = lens2(imSpatialDimensions)
    #    occluderWindow1 = horizontalColumn([2*i-1 for i in imSpatialDimensions], 1/4)
        occluderWindow2 = pinCircle([2*i-1 for i in imSpatialDimensions], 1/8)
#        occluderWindow2 = verticalColumn([2*i-1 for i in imSpatialDimensions], 1/4)
    #    occluderWindow = spectrallyFlatOccluder(imSpatialDimensions)

    #    viewRepeatedOccluder(occluderWindow)
        viewSingleOccluder(occluderWindow1)
        viewSingleOccluder(occluderWindow2)

#        bits = getBitsOfOccluder(occluderWindow, beta, snr*100)
#        pixels = bits/log(IM_BRIGHTNESS/THERMAL_NOISE*100 + 1, 2)
#        sideLength = sqrt(pixels)

#        R = bits / (imSpatialDimensions[0]*imSpatialDimensions[1])

#        D = power*exp(-2*R)

#        print "D:", D

#        print bits, pixels, sideLength

#        obsPlaneIm1 = doFuncToEachChannel(convolve2dMaker(occluderWindow1), im)
        obsPlaneIm1 = doFuncToEachChannel(lambda x: \
            vectorizedDot(make2DTransferMatrix(imSpatialDimensions, occluderWindow1), x, \
            imSpatialDimensions), im) 

        obsPlaneIm2 = doFuncToEachChannel(lambda x: \
            vectorizedDot(make2DTransferMatrix(imSpatialDimensions, occluderWindow2), x, \
            imSpatialDimensions), im) 

        transferMatrix1 = make2DTransferMatrix(imSpatialDimensions, occluderWindow1)
        transferMatrix2 = make2DTransferMatrix(imSpatialDimensions, occluderWindow2)

#        p.matshow(transferMatrix1)
#        p.show()
#        p.matshow(transferMatrix2)
#        p.show()

        combinedMatrix = np.concatenate((transferMatrix1, transferMatrix2), 0)

#        p.matshow(combinedMatrix)
#        p.show()

        viewFrame(obsPlaneIm1, adaptiveScaling=True)
        viewFrame(obsPlaneIm2, adaptiveScaling=True)

        obsPlaneIm2 = doFuncToEachChannel(convolve2dMaker(occluderWindow2), im)

        combinedObsPlaneIm = doFuncToEachChannel(lambda x: \
            vectorizedDot(combinedMatrix, x, \
            (imSpatialDimensions[0]*2, imSpatialDimensions[1])), im) 

        viewFrame(combinedObsPlaneIm, adaptiveScaling=True)

    #    viewFrame(obsPlaneIm, 3e-10)

        noisyObsPlane = addNoise(combinedObsPlaneIm, THERMAL_NOISE)

        inverseMatrix = getPseudoInverse(combinedMatrix, imSpatialDimensions, \
            snr, beta)

#        p.matshow(np.real(inverseMatrix))
#        p.show()

    #    recoveryWindowSimple = getRecoveryWindowSimple(occluderWindow)
    #    recoveredImSimple = doFuncToEachChannel(convolve2dMaker(recoveryWindowSimple), \
    #        noisyObsPlaneIm)

    #    recoveryWindowSophisticated = getRecoveryWindowSophisticated(occluderWindow, beta, snr)
        recoveredIm = doFuncToEachChannel(lambda x: \
            vectorizedDot(inverseMatrix, x, imSpatialDimensions), noisyObsPlane)

        viewFrame(recoveredIm, adaptiveScaling=True)


    #    print THERMAL_NOISE * np.ones(im.shape)

    #    print recoveredIm

    #    recoveredIm -= THERMAL_NOISE * np.ones(im.shape)

    #    print recoveredIm

    #    viewFrame(obsPlaneIm, 2e-10)
    #    viewFrame(recoveredImSimple, 5e-11)
        print "recovered image: lower right"
    #    viewFrame(recoveredImSophisticated, 9e-11)

    #    averageObsPlaneIntensity = sum(sum(noisyObsPlaneIm, 0), 0)/numPixels
    #    averageObsPlane = np.kron(np.ones(imSpatialDimensions), averageObsPlaneIntensity)

    #    print averageObsPlane.shape


    #    print recoveredImSophisticated, 3*noisyObsPlaneIm

    #    print recoveredImSophisticated - 3*noisyObsPlaneIm

        pickle.dump(recoveredImSophisticated, open("recovered_im_2.p", "w"))

        viewFrame(recoveredImSophisticated, adaptiveScaling=True)
    #    viewFrame(recoveredImSophisticated, 1e-10, differenceImage=True, meanSubtraction=True)
    #    viewFrame(recoveredImSophisticated, 1e-9, differenceImage=True, meanSubtraction=True)

    #    viewFrame(recoveredImSophisticated, 1e-10, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)
    #    viewFrame(recoveredImSophisticated, 1e-9, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)
    #    viewFrame(recoveredImSophisticated, 1e-8, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)


    #    viewFrame(recoveredImSophisticated - 3*noisyObsPlaneIm, 1e-10, differenceImage=True)
    #    viewFrame(recoveredImSophisticated - 3*noisyObsPlaneIm, 1e-8, differenceImage=True)

    #    viewFrame(recoveredImSophisticated, 6e-11)
    #    viewFrame(recoveredImSophisticated, 2e-11)

    #    viewFrame(recoveredImSophisticated, 3e-9)


        diff = im - recoveredImSophisticated
    #    diff = averageVertically(im) - recoveredImSophisticated
    #    print diff

        print im
        print recoveredImSophisticated

        empD = np.real(np.sum(np.sum(np.sum(np.multiply(diff, diff), 0), 0), 0)/(numPixels*3))

        print "empirical D:", empD*1e-20

        print "reconstruction SNR:", 10*log(power/empD, 10)

    #    viewFrame(recoveredImSophisticated, 3e-9)

    if JOIN_INFO_BY_LIST:
    #    imRaw = Image.open("winnie.png")
    #    im = np.array(imRaw).astype(float)

        IM_BRIGHTNESS = 1e10
        THERMAL_NOISE = 1e6
    #    THERMAL_NOISE = 1

        beta = 0.1
    #    beta = 1

    #    snr = IM_BRIGHTNESS / THERMAL_NOISE

    #    snr = 1e4
    #    snr = 1e5
        snr = 3e6
    #    snr = 1e12

    #    im = pickle.load(open("winnie_downsampled_a_lot.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("calvin_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_slightly_downsampled.p", "r"))*IM_BRIGHTNESS
    #    im = pickle.load(open("winnie_clipped.p", "r"))*IM_BRIGHTNESS
        im = pickle.load(open("dora_very_downsampled.p", "r"))*IM_BRIGHTNESS

        imSpatialDimensions = im.shape[:2]

        numPixels = imSpatialDimensions[0]*imSpatialDimensions[1]

        power = np.sum(np.sum(np.sum(np.multiply(im, im), 0), 0), 0)/(numPixels*3)

        print "power", power

        #print im.shape

        print imSpatialDimensions

    #    im = makeImpulseImage(imSpatialDimensions, "center")*IM_BRIGHTNESS
    #    im = makeSquareImage(imSpatialDimensions, 1/4)*IM_BRIGHTNESS

        viewFrame(im, 1e-10)
    #    viewFrame(im, 1e-10, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)
    #    viewFrame(np.flip(np.flip(im, 0), 1), 1e-10)
    #    viewFrame(averageVertically(im), 1e-10)


        forwardOccluder1 = elephantOccluder([2*i-1 for i in imSpatialDimensions], 1/2)
        forwardOccluder2 = verticalColumn([2*i-1 for i in imSpatialDimensions], 1/4)
        forwardOccluder3 = horizontalColumn([2*i-1 for i in imSpatialDimensions], 1/4)

    #    backwardOccluder1 = pinSquare([2*i-1 for i in imSpatialDimensions], 1/4)
        backwardOccluder1 = elephantOccluder([2*i-1 for i in imSpatialDimensions], 1/2)
#        backwardOccluder1 = pinCircle([2*i-1 for i in imSpatialDimensions], 1/6)
        backwardOccluder2 = verticalColumn([2*i-1 for i in imSpatialDimensions], 1/4)
        backwardOccluder3 = horizontalColumn([2*i-1 for i in imSpatialDimensions], 1/4)    #    occluderWindow1 = pinCircle([2*i-1 for i in imSpatialDimensions], 1/8)
    #    occluderWindow = circleSpeck(imSpatialDimensions, 1/4)
    #    occluderWindow = coarseCheckerBoard(imSpatialDimensions)
    #    occluderWindow = fineCheckerBoard(imSpatialDimensions)
    #    occluderWindow = pinSquare(imSpatialDimensions, 1/4)
    #    occluderWindow2 = squareSpeck([2*i-1 for i in imSpatialDimensions], 1/4)
    #    occluderWindow = pinHole(imSpatialDimensions)
    #    occluderWindow = pinSpeck(imSpatialDimensions)
    #    occluderWindow = randomOccluder(imSpatialDimensions)
    #    occluderWindow = lens2(imSpatialDimensions)
#        occluderWindow2 = pinCircle([2*i-1 for i in imSpatialDimensions], 1/8)
    #    occluderWindow2 = verticalColumn([2*i-1 for i in imSpatialDimensions], 1/4)
    #    occluderWindow3 = horizontalColumn([2*i-1 for i in imSpatialDimensions], 1/4)
    #    occluderWindow = spectrallyFlatOccluder(imSpatialDimensions)

    #    viewRepeatedOccluder(occluderWindow)
 #       viewSingleOccluder(occluderWindow1)
#        viewSingleOccluder(occluderWindow2)
#        viewSingleOccluder(occluderWindow3)

        print np.sum(forwardOccluder1)
        print np.sum(backwardOccluder1)

        viewSingleOccluder(forwardOccluder1)
        viewSingleOccluder(backwardOccluder1)

#        listOfOccluders = [occluderWindow1]
        listOfForwardOccluders = [forwardOccluder1, forwardOccluder2, forwardOccluder3]
        listOfBackwardOccluders = [backwardOccluder1, backwardOccluder2, backwardOccluder3]

#        listOfForwardOccluders = [forwardOccluder1]
#        listOfBackwardOccluders = [backwardOccluder1]

 #       listOfOccluders = [occluderWindow1, occluderWindow2, occluderWindow3]
#        bits = getBitsOfOccluder(occluderWindow, beta, snr*100)
#        pixels = bits/log(IM_BRIGHTNESS/THERMAL_NOISE*100 + 1, 2)
#        sideLength = sqrt(pixels)

#        R = bits / (imSpatialDimensions[0]*imSpatialDimensions[1])

#        D = power*exp(-2*R)

#        print "D:", D

#        print bits, pixels, sideLength

#        obsPlaneIm1 = doFuncToEachChannel(convolve2dMaker(occluderWindow1), im)
#        obsPlaneIm1 = doFuncToEachChannel(lambda x: \
#            vectorizedDot(make2DTransferMatrix(imSpatialDimensions, occluderWindow1), x, \
#            imSpatialDimensions), im) 

#        obsPlaneIm2 = doFuncToEachChannel(lambda x: \
#            vectorizedDot(make2DTransferMatrix(imSpatialDimensions, occluderWindow2), x, \
#            imSpatialDimensions), im) 

#        transferMatrix1 = make2DTransferMatrix(imSpatialDimensions, occluderWindow1)
#        transferMatrix2 = make2DTransferMatrix(imSpatialDimensions, occluderWindow2)

        listOfForwardTransferMatrices = [make2DTransferMatrix(imSpatialDimensions, occ) \
            for occ in listOfForwardOccluders]
        combinedForwardMatrix = np.concatenate(listOfForwardTransferMatrices, 0)

        listOfBackwardTransferMatrices = [make2DTransferMatrix(imSpatialDimensions, occ) \
            for occ in listOfBackwardOccluders]
        combinedBackwardMatrix = np.concatenate(listOfBackwardTransferMatrices, 0)

#        p.matshow(transferMatrix1)
#        p.show()
#        p.matshow(transferMatrix2)
#        p.show()

#        p.matshow(combinedMatrix)
#        p.show()

#        viewFrame(obsPlaneIm1, adaptiveScaling=True)
#        viewFrame(obsPlaneIm2, adaptiveScaling=True)

#        obsPlaneIm2 = doFuncToEachChannel(convolve2dMaker(occluderWindow2), im)
    
        numOccluders = len(listOfForwardOccluders)
    
        combinedObsPlaneIm = doFuncToEachChannel(lambda x: \
            vectorizedDot(combinedForwardMatrix, x, \
            (imSpatialDimensions[0]*numOccluders, imSpatialDimensions[1])), im) 

        viewFrame(combinedObsPlaneIm, adaptiveScaling=True)

    #    viewFrame(obsPlaneIm, 3e-10)

        noisyObsPlane = addNoise(combinedObsPlaneIm, THERMAL_NOISE)

        inverseMatrix = getPseudoInverse(combinedBackwardMatrix, imSpatialDimensions, \
            snr, beta)

#        p.matshow(np.real(inverseMatrix))
#        p.show()

    #    recoveryWindowSimple = getRecoveryWindowSimple(occluderWindow)
    #    recoveredImSimple = doFuncToEachChannel(convolve2dMaker(recoveryWindowSimple), \
    #        noisyObsPlaneIm)

    #    recoveryWindowSophisticated = getRecoveryWindowSophisticated(occluderWindow, beta, snr)
        recoveredIm = doFuncToEachChannel(lambda x: \
            vectorizedDot(inverseMatrix, x, imSpatialDimensions), noisyObsPlane)

        viewFrame(recoveredIm, adaptiveScaling=True, magnification=1.5)


    #    print THERMAL_NOISE * np.ones(im.shape)

    #    print recoveredIm

    #    recoveredIm -= THERMAL_NOISE * np.ones(im.shape)

    #    print recoveredIm

    #    viewFrame(obsPlaneIm, 2e-10)
    #    viewFrame(recoveredImSimple, 5e-11)
        print "recovered image: lower right"
    #    viewFrame(recoveredImSophisticated, 9e-11)

    #    averageObsPlaneIntensity = sum(sum(noisyObsPlaneIm, 0), 0)/numPixels
    #    averageObsPlane = np.kron(np.ones(imSpatialDimensions), averageObsPlaneIntensity)

    #    print averageObsPlane.shape


    #    print recoveredImSophisticated, 3*noisyObsPlaneIm

    #    print recoveredImSophisticated - 3*noisyObsPlaneIm

        pickle.dump(recoveredImSophisticated, open("recovered_im_2.p", "w"))

        viewFrame(recoveredImSophisticated, magnification=2.5, adaptiveScaling=True)
    #    viewFrame(recoveredImSophisticated, 1e-10, differenceImage=True, meanSubtraction=True)
    #    viewFrame(recoveredImSophisticated, 1e-9, differenceImage=True, meanSubtraction=True)

    #    viewFrame(recoveredImSophisticated, 1e-10, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)
    #    viewFrame(recoveredImSophisticated, 1e-9, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)
    #    viewFrame(recoveredImSophisticated, 1e-8, differenceImage=False, meanSubtraction=True,
    #        absoluteMeanSubtraction=True)


    #    viewFrame(recoveredImSophisticated - 3*noisyObsPlaneIm, 1e-10, differenceImage=True)
    #    viewFrame(recoveredImSophisticated - 3*noisyObsPlaneIm, 1e-8, differenceImage=True)

    #    viewFrame(recoveredImSophisticated, 6e-11)
    #    viewFrame(recoveredImSophisticated, 2e-11)

    #    viewFrame(recoveredImSophisticated, 3e-9)


        diff = im - recoveredImSophisticated
    #    diff = averageVertically(im) - recoveredImSophisticated
    #    print diff

        print im
        print recoveredImSophisticated

        empD = np.real(np.sum(np.sum(np.sum(np.multiply(diff, diff), 0), 0), 0)/(numPixels*3))

        print "empirical D:", empD*1e-20

        print "reconstruction SNR:", 10*log(power/empD, 10)

    #    viewFrame(recoveredImSophisticated, 3e-9)

    if PROJECTION_INFO:
        vid = pickle.load(open("steven_batched.p", "r"))

        imSpatialDimensions = vid[0].shape[:-1]

        beta = 0.0001
        snr = 0.001

        v = verticalTransferMatrix(imSpatialDimensions)
        h = horizontalTransferMatrix(imSpatialDimensions)

        listOfForwardTransferMatrices = [v, h]
        combinedForwardMatrix = np.concatenate(listOfForwardTransferMatrices, 0)

        listOfBackwardTransferMatrices = [v, h]
        combinedBackwardMatrix = np.concatenate(listOfBackwardTransferMatrices, 0)

        inverseMatrix = getPseudoInverse(combinedBackwardMatrix, imSpatialDimensions, \
            snr, beta)

        for i, frame in enumerate(vid):
            print i

            obs = doFuncToEachChannel(lambda x: vectorizedDot(combinedForwardMatrix, x, \
                (50, \
                len(listOfBackwardTransferMatrices))), frame)

            recon = doFuncToEachChannel(lambda x: vectorizedDot(inverseMatrix, x, \
                imSpatialDimensions), obs)

            viewFrame(recon, adaptiveScaling=True, filename="proj_movie/frame_" + \
                padIntegerWithZeros(i, 3) + ".png")

    if EQUIV_OCC:

        im = pickle.load(open("dora_very_downsampled.p", "r"))

        viewFrame(im, magnification=0.5, differenceImage=True)

        print im.shape

        imDims = im.shape[:-1]
        occDims = [2*i-1 for i in imDims]

        snr = 1e10
        beta = 1

        occ1 = cornerOccluder(occDims)

#        occ2 = np.flip(occ1, 1)
        occ2 = cornerAperture(occDims)
        occ2 = np.flip(occ2, 1)
#        occ2 = pinSquare(occDims, 1/5)
#        occ2 = twoSquares(occDims, 1/5)
#        occ2 = cornerCheckerboard(occDims, reverse=True)


        viewFrame(imageify(occ1), adaptiveScaling=True)
        viewFrame(imageify(occ2), adaptiveScaling=True)

        transferMat1 = make2DTransferMatrix(imDims, occ1)
        transferMat2 = make2DTransferMatrix(imDims, occ2)

#        inverseMatrix = getPseudoInverse(transferMat2, imDims, \
#            snr, beta)

        inverseMatrix = snr*np.linalg.inv(snr*transferMat2 + \
            np.identity(transferMat2.shape[0]))

        doraObs = doFuncToEachChannel(lambda x: vectorizedDot(transferMat1, x, imDims), im)

        viewFrame(doraObs, adaptiveScaling=True)

        recon = doFuncToEachChannel(lambda x: vectorizedDot(inverseMatrix, x, imDims), doraObs)

        print recon

        viewFrame(recon, adaptiveScaling=False, magnification=0.5, differenceImage=True)

    if EQUIV_OCC_2:

        im = pickle.load(open("dora_very_downsampled.p", "r"))

        viewFrame(im, magnification=0.5, differenceImage=True)

        print im.shape

        imDims = im.shape[:-1]
        occDims = [2*i-1 for i in imDims]

        snr = 1e10
        beta = 1

        derivativeArray = np.zeros(occDims)

        nibLoc = (int(occDims[0]/2), int(occDims[1]/2))

        highVal = 1/(occDims[0]*occDims[1])

        derivativeArray[nibLoc[0],nibLoc[1]] = highVal

        verticalConstants1 = np.zeros(occDims[1])
        horizontalConstants1 = np.zeros(occDims[0])

#        verticalConstants1[nibLoc[1]] = 0

        verticalConstants2 = np.zeros(occDims[1])
        horizontalConstants2 = np.zeros(occDims[0])
        verticalConstants2[nibLoc[1]] = -highVal
#        horizontalConstants2 = highVal*np.ones(occDims[0])

        occ1 = occluderGenerator(derivativeArray, verticalConstants1, horizontalConstants1) 

        print occ1.shape
        viewFrame(imageify(occ1), adaptiveScaling=True)

        print horizontalConstants2

        occ2 = occluderGenerator(derivativeArray, verticalConstants2, horizontalConstants2) 

        viewFrame(imageify(occ2), adaptiveScaling=True)

        transferMat1 = make2DTransferMatrix(imDims, occ1)
        transferMat2 = make2DTransferMatrix(imDims, occ2)

#        inverseMatrix = getPseudoInverse(transferMat2, imDims, \
#            snr, beta)

        inverseMatrix = snr*np.linalg.inv(snr*transferMat2 + \
            np.identity(transferMat2.shape[0]))

        doraObs = doFuncToEachChannel(lambda x: vectorizedDot(transferMat1, x, imDims), im)

        viewFrame(doraObs, adaptiveScaling=True)

        recon = doFuncToEachChannel(lambda x: vectorizedDot(inverseMatrix, x, imDims), doraObs)

        print recon

        viewFrame(recon, adaptiveScaling=False, magnification=0.5, differenceImage=True)

    if EQUIV_OCC_3:

        im = pickle.load(open("dora_very_downsampled.p", "r"))

        viewFrame(im, magnification=0.5, differenceImage=True)

        print im.shape

        imDims = im.shape[:-1]
        occDims = [2*i-1 for i in imDims]

        snr = 1e10
        beta = 1

        derivativeArray = np.zeros(occDims)

        nib1X = int(occDims[0]/3)
        nib2X = int(2*occDims[0]/3)

        nib1Y = int(occDims[1]/3)
        nib2Y = int(2*occDims[1]/3)

        highVal = 1/(occDims[0]*occDims[1])

        derivativeArray[nib1X,nib1Y] = highVal
        derivativeArray[nib1X,nib2Y] = -highVal
        derivativeArray[nib2X,nib1Y] = -highVal
        derivativeArray[nib2X,nib2Y] = highVal

        verticalConstants1 = np.zeros(occDims[1])
        horizontalConstants1 = np.zeros(occDims[0])

#        verticalConstants1[nibLoc[1]] = 0

        verticalConstants2 = np.zeros(occDims[1])
        horizontalConstants2 = np.zeros(occDims[0])
        horizontalConstants2 = highVal*np.ones(occDims[0])

        verticalConstants2[nib1Y] = -highVal
        verticalConstants2[nib2Y] = highVal

        occ1 = occluderGenerator(derivativeArray, verticalConstants1, horizontalConstants1) 

        print occ1.shape
        viewFrame(imageify(occ1), adaptiveScaling=True)

        occ2 = occluderGenerator(derivativeArray, verticalConstants2, horizontalConstants2) 

        viewFrame(imageify(occ2), adaptiveScaling=True)

        transferMat1 = make2DTransferMatrix(imDims, occ1)
        transferMat2 = make2DTransferMatrix(imDims, occ2)

#        inverseMatrix = getPseudoInverse(transferMat2, imDims, \
#            snr, beta)

        inverseMatrix = snr*np.linalg.inv(snr*transferMat2 + \
            np.identity(transferMat2.shape[0]))

        doraObs = doFuncToEachChannel(lambda x: vectorizedDot(transferMat1, x, imDims), im)

        viewFrame(doraObs, adaptiveScaling=True)

        recon = doFuncToEachChannel(lambda x: vectorizedDot(inverseMatrix, x, imDims), doraObs)

        print recon

        viewFrame(recon, adaptiveScaling=False, magnification=0.5, differenceImage=True)

    if EQUIV_OCC_MOVIE_1:

        im = pickle.load(open("dora_very_downsampled.p", "r"))

        viewFrame(im, magnification=0.5, differenceImage=True)

        print im.shape

        imDims = im.shape[:-1]
        occDims = [2*i-1 for i in imDims]

        snr = 1e10
        beta = 1

        occ1 = cornerOccluder(occDims)
        transferMat1 = make2DTransferMatrix(imDims, occ1)

        doraObs = doFuncToEachChannel(lambda x: vectorizedDot(transferMat1, x, imDims), im)

        frameCounter = 0

        for cornerLocX in range(0, occDims[0], 12):
            for cornerLocY in range(0, occDims[1], 12):
                print frameCounter

                cornerLoc = (cornerLocX, cornerLocY)

                occ2 = cornerOccluderMovingCenter(cornerLoc, occDims)

                transferMat2 = make2DTransferMatrix(imDims, occ2)

#                inverseMatrix = getPseudoInverse(transferMat2, imDims, \
#                    snr, beta)

                inverseMatrix = snr*np.linalg.inv(snr*transferMat2 + \
                    np.identity(transferMat2.shape[0]))

#        viewFrame(doraObs, adaptiveScaling=True)

                recon = doFuncToEachChannel(lambda x: vectorizedDot(inverseMatrix, x, imDims), doraObs)

#                viewFrame(recon, adaptiveScaling=False, magnification=0.5, differenceImage=True, \
#                    filename="equiv_occ_1/frame_" + padIntegerWithZeros())

                p.clf()

                p.subplot(121)
#                p.axis("off")
                viewFrame(imageify(occ2), filename="pass", adaptiveScaling=True,
                    relax=True, magnification=1, differenceImage=False)

                p.subplot(122)
#                p.axis("off")
                viewFrame(recon, filename="pass", adaptiveScaling=False,
                    relax=True, magnification=0.5, differenceImage=True)

                p.savefig("equiv_occ_2/frame_" + padIntegerWithZeros(frameCounter, 4) + ".png")

                frameCounter += 1

    if EQUIV_OCC_MOVIE_2:

        im = pickle.load(open("dora_very_downsampled.p", "r"))

        viewFrame(im, magnification=0.5, differenceImage=True)

        print im.shape

        imDims = im.shape[:-1]
        occDims = [2*i-1 for i in imDims]

        snr = 1e10
        beta = 1

        occ1 = cornerOccluder(occDims)
        transferMat1 = make2DTransferMatrix(imDims, occ1)

        doraObs = doFuncToEachChannel(lambda x: vectorizedDot(transferMat1, x, imDims), im)

        frameCounter = 0

        for squareRadius in np.linspace(0, 1/2, 30):
            print frameCounter

            occ2 = pinSquare(occDims, squareRadius)

            transferMat2 = make2DTransferMatrix(imDims, occ2)

#                inverseMatrix = getPseudoInverse(transferMat2, imDims, \
#                    snr, beta)

            inverseMatrix = snr*np.linalg.inv(snr*transferMat2 + \
                np.identity(transferMat2.shape[0]))

#        viewFrame(doraObs, adaptiveScaling=True)

            recon = doFuncToEachChannel(lambda x: vectorizedDot(inverseMatrix, x, imDims), doraObs)

#                viewFrame(recon, adaptiveScaling=False, magnification=0.5, differenceImage=True, \
#                    filename="equiv_occ_1/frame_" + padIntegerWithZeros())

            p.clf()

            p.subplot(121)
#                p.axis("off")
            viewFrame(imageify(occ2), filename="pass", adaptiveScaling=True,
                relax=True, magnification=1, differenceImage=False)

            p.subplot(122)
#                p.axis("off")
            viewFrame(recon, filename="pass", adaptiveScaling=False,
                relax=True, magnification=0.5, differenceImage=True)

            p.savefig("equiv_occ_3/frame_" + padIntegerWithZeros(frameCounter, 4) + ".png")

            frameCounter += 1