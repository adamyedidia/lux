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
import imageio
from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import sys
from scipy.linalg import dft
from best_matrix import findAssignment, isPrime, xorProduct, padIntegerWithZeros, \
    getToeplitzLikeTransferMatrixWithVariedDepth
from scipy.linalg import circulant, toeplitz


MATRIX_SIZE_TEST = False
FOURIER_UNDERSTANDER = False
FOURIER_UNDERSTANDER_2D = False
ALRIGHT_LETS_DO_THIS = False
TOEPLITZ = False
TOEPLITZ_2D = False
ANTONIO_METHOD = False
DIFFERENT_DEPTHS_SIM = False
ACTIVE_SIM = True

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

def oneDPinHole(length, widthOfPinhole):
    return [1/length*(abs(i - length/2) < length*widthOfPinhole/2) for i in range(length)]

def oneDRandom(length):
    return [1/length*(random.random() > 0.5) for _ in range(length)]

def oneDedge(length):
    return [0]*int((length-1)/2)+[1/length]*int((length+1)/2)

def oneDNothing(length):
    return [1/length for _ in range(length)]

def verticalColumn(imShape, width):
    print "column"

    midPointY = int(imShape[1]/2)

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



def pinCircle(imShape, radius):
    print "circle"
    midPointX = int(imShape[0]/2)
    midPointY = int(imShape[1]/2)

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

def viewSingleOccluder(occluderWindow, filename):
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

                        matColumn.append(attenuationFactor)
                    else:
                        matColumn.append(0)

            returnMat.append(matColumn)

    return np.transpose(np.array(returnMat))

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
    THERMAL_NOISE = 1e8
#    THERMAL_NOISE = 1


    beta = 0.3
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


    occluderWindow = pinCircle(imSpatialDimensions, 1/8)
#    occluderWindow = circleSpeck(imSpatialDimensions, 1/4)
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

    recoveryWindowSophisticated = recoveryWindowSophisticated(occluderWindow, beta, snr)
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

    viewFrame(recoveredImSophisticated, 6e-11)
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
    obsThetaRes = obsXRes
    obsTRes = 40

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

#    transferMatrix = buildActiveTransferMatrixWithCorner(sceneXRes, sceneYRes, \
#        obsThetaRes, obsTRes, x, y, r)

    numTransitions = 25

    randomOccluderFunc = randomOccluderFuncMaker(numTransitions)

    transferMatrix = buildActiveTransferMatrixWithOccluder(illuminationX, sceneXRes, sceneYRes, \
        obsXRes, obsTRes, y1, y2, x1, randomOccluderFunc)



#    p.matshow(transferMatrix)
#    p.show()

#    p.matshow(transferMatrix)
#    p.colorbar()
#    p.show()

    displayOccluderFunc(randomOccluderFunc)


    obsPlane = doFuncToEachChannelVec(lambda x: np.dot(transferMatrix, x), imVector)

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
