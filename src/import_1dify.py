from __future__ import division
import numpy as np
import pylab
from math import pi, sqrt, ceil, floor
import matplotlib.pyplot as p
from PIL import Image
from PIL import ImageFilter
import pickle
from process_image import ungarbleImageX, ungarbleImageY, \
    createGarbleMatrixX, createGarbleMatrixY, ungarbleImageXOld, \
    ungarbleImageYOld, getQ
import imageio
from video_magnifier import viewFrame, viewFrameR
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import sys
import random
from sklearn import linear_model

#np.set_printoptions(threshold=np.nan)

EPS = 1e-8

derivativeOfMovie = False
exportFlatVideo = False
displayFrameOfVideo = False
displayFlatFrameOfVideo = False
buildVideos = False
findOccluderWithSparsity = False
findOccluderWithSparsityEntireHistory = False
fourierApproach = False
exportFlatRealVideo = False
showArtificialOccluder = False
testDifferentDepthsOccluder = False
getOutMovie = False
readMovie = False
displaySimpleOccluder = False
polynomialApproach = False


REAL_OCCLUDER_ARRAY = [1]*44 +[0]*26 + [1]*113 + [0]*58 + [1]*5 + [0]*110 + \
    [1]*40 + [0]*50



def verticalTransferMatrix(n):
    transferMatrixList = []
    for i in range(n):
        row = int(floor(n/4))*[1] + int(floor(i/2))*[1] + 3*[0]
#            int(floor(n/2-i/2-5))*[1] + int(floor(n/4))*[1]

#        print len(row)

        row += [1]*(n-len(row))

#        print i, row

        transferMatrixList.append(row)



#    print np.array(transferMatrixList)

    return np.array(transferMatrixList)


def horizontalTransferMatrix(n):
    return np.transpose(verticalTransferMatrix(n))

def differentDepthsTransferMatrix(n):
    return np.minimum(verticalTransferMatrix(n),horizontalTransferMatrix(n))

def dVideoDT(listOfFlatFrames):
    return np.array([listOfFlatFrames[i+1] - listOfFlatFrames[i] for i in \
        range(len(listOfFlatFrames)-1)])

def turnRealFrameIntoFlatFrame(realFrame, leftBottomPixel, \
    rightBottomPixel):

    frameShape = realFrame.shape
    frameHeight = frameShape[0]

    pixelLinspace = np.linspace(leftBottomPixel, rightBottomPixel, \
        rightBottomPixel)

    flatFrameWidth = rightBottomPixel - leftBottomPixel + 1
    currentArray = np.array([[0., 0., 0.]]*flatFrameWidth)

    for rowNum in range(frameHeight):
        currentArray += getRowOfRealFrame(realFrame, leftBottomPixel, \
            rightBottomPixel, rowNum)

    currentArray /= frameHeight

    return currentArray

def getRowOfRealFrame(realFrame, leftBottomPixel, rightBottomPixel, \
    rowNum):

    frameShape = realFrame.shape
    frameHeight = frameShape[0]
    frameWidth = frameShape[1]

#    print rowNum, "/", frameHeight

    rawRow = realFrame[rowNum]

    effectiveLeftPixel = (rowNum/frameHeight)*leftBottomPixel
    effectiveRightPixel = (rowNum/frameHeight)*rightBottomPixel + \
        (1-rowNum/frameHeight)*(frameWidth-1)

#    print np.linspace(effectiveLeftPixel, \
#        effectiveRightPixel, rightBottomPixel - leftBottomPixel + 1)

    return [fuzzyLookup(rawRow, i) for i in np.linspace(effectiveLeftPixel,
        effectiveRightPixel, rightBottomPixel - leftBottomPixel + 1)]

def average(x):
    return sum(x)/len(x)

def turnVideoIntoListOfFlattenedFrames(vid, firstFrame=0, lastFrame=None):
    listOfFlattenedFrames = []
    numFrames = len(vid)

    if lastFrame == None:
        lastFrame = numFrames

    for i in range(firstFrame, lastFrame): #range(int(numFrames)): #i in range(400, 500):
        print i, "/", numFrames
        im = vid.get_data(i)
        frame = np.array(im).astype(float)
        listOfFlattenedFrames.append(flattenFrame(frame))

    return listOfFlattenedFrames

def batchList(listOfFrames, batchSize):
    return np.array([average(i) for i in np.array_split(listOfFrames, \
        int(len(listOfFrames)/batchSize))])

# index is a float
def fuzzyLookup(array, index):
#    print len(array), index

    floorIndex = int(floor(index))
    ceilIndex = int(ceil(index))

    residue = index % 1

    arrayBelow = array[floorIndex]
    arrayAbove = array[ceilIndex]

    return (1-residue) * arrayBelow + residue * arrayAbove
    
def fuzzyLookup2D(array, i, j):
    floorI = int(floor(i))
    ceilI = int(ceil(i))
    floorJ = int(floor(j))
    ceilJ = int(ceil(j))
    
    resI = i % 1
    resJ = j % 1
    
    fIfJ = array[floorI][floorJ]
    fIcJ = array[floorI][ceilJ]
    cIfJ = array[ceilI][floorJ]
    cIcJ = array[ceilI][ceilJ]

#    print fIfJ, fIcJ, cIfJ, cIcJ
    
    return (1-resI)*(1-resJ)*fIfJ + \
        (1-resI)*resJ*fIcJ + \
        resI*(1-resJ)*cIfJ + \
        resI*resJ*cIcJ

def turnRealMovieIntoListOfFlatFrames(listOfFrames, batchSize):
    pass

def getAverageBrightnessOfFlatFrame(flatFrame):
    return average([average(i) for i in flatFrame])

def getFlatFrameOccluderPattern(flatFrame):
    avBrightness = getAverageBrightnessOfFlatFrame(flatFrame)
    tentativeOccluderPattern = np.array([1*(average(i)>avBrightness) for i in flatFrame])

    if tentativeOccluderPattern[0] == 0:
        return tentativeOccluderPattern
    else:
        return np.array([1-x for x in tentativeOccluderPattern])

def takeXDerivativeOfFlatFrame(flatFrame):
#    print flatFrame
    dFlatFrameDX = [flatFrame[i+1] - flatFrame[i] for \
        i in range(len(flatFrame) - 1)]
#    print dFlatFrameDX

    return np.array(dFlatFrameDX)

def findTransitionsInFlatFrame(flatFrame):
    largeDerivativeThreshold = 1

    transitions = []
    isRecentTransition = False

    dFlatFrameDX = takeXDerivativeOfFlatFrame(flatFrame)
    for i in range(len(dFlatFrameDX)):
        print np.linalg.norm(dFlatFrameDX[i]), \
            np.linalg.norm(dFlatFrameDX[i]) > largeDerivativeThreshold, \
            isRecentTransition,

        if (np.linalg.norm(dFlatFrameDX[i]) > largeDerivativeThreshold) and \
            not isRecentTransition:

            transitions.append(1)
            isRecentTransition = True

        elif (np.linalg.norm(dFlatFrameDX[i]) > largeDerivativeThreshold) and \
            isRecentTransition:

            transitions.append(0)

        else:
            transitions.append(0)
            isRecentTransition = False

    return transitions


def blackAndWhiteifyFlatFrame(flatFrame):
    return np.swapaxes(np.array([flatFrame]*3),0,1)

def flattenFrame(frame):
    rearrangedFrame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)
    outOfOrderFlattenedFrame = np.array([average(i) for i in rearrangedFrame])
    return np.swapaxes(outOfOrderFlattenedFrame, 0, 1)

def displayFlattenedFrame(flattenedFrame, height, magnification=1, \
    differenceImage=False, filename=None):

    frame = np.array([flattenedFrame]*height)
    viewFrame(frame, differenceImage=differenceImage, magnification=magnification,
        filename=filename)

def displayMultipleFlatFrames(listOfFlatFrames, height, magnification=1, \
    differenceImage=False, filename=None):

    listOfStretchedFrames = [np.array([flattenedFrame]*height) for flattenedFrame \
        in listOfFlatFrames]

def allListsOfSizeX(x):
    if x == 0:
        return []
    else:
        return [[0] + i for i in allListsOfSizeX(x-1)] + \
            [[1] + i for i in allListsOfSizeX(x-1)]

def stretchOccluderArray(smallOccluderArray, newLength):
    numRepeats = int(ceil(newLength / len(smallOccluderArray)))
    untrimmedOccluderArray = np.repeat(smallOccluderArray, numRepeats)
    return untrimmedOccluderArray[:newLength]

def stretchArray(arr, newShape):
    oldShape = arr.shape

    newArray = []
    for i in np.linspace(0, oldShape[0]-1, newShape[0]):
        newArray.append([])
        for j in np.linspace(0, oldShape[1]-1, newShape[1]):
            newArray[-1].append(fuzzyLookup2D(arr, i, j))

    return np.array(newArray)

def generateRandomOccluder(length, pSwitch):
    currentValue = 1

    occluderList = []
    for i in range(length):
        if random.random() < pSwitch:
            currentValue = 1 - currentValue

        occluderList.append(currentValue)

    return np.array(occluderList)

def generateRandomOccluderList(length, pSwitch):
    currentValue = 1

    occluderList = []
    for i in range(length):
        if random.random() < pSwitch:
            currentValue = 1 - currentValue

        occluderList.append(currentValue)

    return occluderList

def getTransferMatrixFromOccluder(occluderArray, slidingWindowSize):
    transferMatrix = []

    for i in range(len(occluderArray) - slidingWindowSize + 1):
        transferMatrix.append(occluderArray[i:i+slidingWindowSize])

    return np.array(transferMatrix)

def getTransferMatrixFromOccluderLong(occluderArray, slidingWindowSize):
    longOccluderArray = [0]*(slidingWindowSize-1) + occluderArray.tolist() + \
        [0]*(slidingWindowSize-1)

    return getTransferMatrixFromOccluder(longOccluderArray, slidingWindowSize)

def recoverScene(transferMatrix, observationVector, alpha=1.):
#    print "trans", recoveredTransferMatrix[int(1.5*flatFrameLength)]

#    print "obs", observationVector

    clf = linear_model.Lasso(alpha=alpha, precompute="auto")
    clf.fit(transferMatrix, observationVector)
    print clf.coef_
    return clf.coef_

def displayConcatenatedArray(arr, rowsPerFrame=10, magnification=1, \
    differenceImage=False, filename=None, stretchFactor=1):

#    print np.shape(arr)

    arrSplit = np.split(arr, np.shape(arr)[1]/3, 1)

#    print len(arrSplit)

    newArrList = []

    
    for frame in arrSplit:
#        print frame.shape

        for _ in range(rowsPerFrame):
            newArrList.append(np.repeat(frame, stretchFactor))

#    print np.array(newArrList).shape

#    print np.array(newArrList).shape

    viewFrame(np.array(newArrList), magnification=magnification, \
        differenceImage=differenceImage, filename=filename)

def getDiffObsFromFrameNum(transferMatrix, frameNum):
    flatFrame1 = listOfFlatFrames[frameNum]
    flatFrame2 = listOfFlatFrames[frameNum+1]

    return np.dot(transferMatrix, flatFrame2 - flatFrame1)

def getDiffFromFrameNum(frameNum):
    flatFrame1 = listOfFlatFrames[frameNum]
    flatFrame2 = listOfFlatFrames[frameNum+1]

    return flatFrame2 - flatFrame1

#def getDifferenceImage(frame1, frame2):

if derivativeOfMovie:
    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)

    flatFrameLength = len(listOfFlatFrames[0])

    occluderArray = generateRandomOccluder(2*flatFrameLength-1, 0.005)

    transferMatrix = getTransferMatrixFromOccluderLong(occluderArray, flatFrameLength)
    transferMatrixShort = getTransferMatrixFromOccluder(occluderArray, flatFrameLength)


#    randomFrameIndex = random.randint(0, len(listOfFlatFrames)-2)
    randomFrameIndex = 153

    diffScene = getDiffFromFrameNum(randomFrameIndex)
    diffSceneObs1 = getDiffObsFromFrameNum(transferMatrix, randomFrameIndex)
    occluderPattern = getFlatFrameOccluderPattern(diffSceneObs1)
    occluderPatternFirst1Index = occluderPattern.tolist().index(1)

    diffSceneObsShort = getDiffObsFromFrameNum(transferMatrixShort, randomFrameIndex)

    prunedOccluderPattern = occluderPattern[occluderPatternFirst1Index: \
        occluderPatternFirst1Index + 2*flatFrameLength-1]


    recoveredTransferMatrixShort = getTransferMatrixFromOccluder(prunedOccluderPattern, flatFrameLength)
#    recoveredTransferMatrixShort = getTransferMatrixFromOccluder(stretchOccluderArray(occluderPattern, \
#        2*flatFrameLength-1), flatFrameLength)
#    recoveredTransferMatrix = getTransferMatrixFromOccluder(occluderArray, flatFrameLength)


    recoveredSceneWithIncorrectOccluder = np.swapaxes(recoverScene(recoveredTransferMatrixShort, \
        diffSceneObsShort, alpha=10.), 0,1)

#    print recoveredSceneWithIncorrectOccluder

    displayFlattenedFrame(diffScene, 100, magnification=10, differenceImage=True)

    displayFlattenedFrame(blackAndWhiteifyFlatFrame(occluderArray), 200,\
        magnification=255, differenceImage=False)

    displayFlattenedFrame(diffSceneObs1, 200, magnification=1, differenceImage=True)

    displayFlattenedFrame(blackAndWhiteifyFlatFrame(np.array(prunedOccluderPattern)), 200,\
        magnification=255, differenceImage=False)

    displayFlattenedFrame(recoveredSceneWithIncorrectOccluder, 100, magnification=10, \
        differenceImage=True)

#    transferMatrix = getTransferMatrixFromOccluderLong(occluderArray, flatFrameLength)

#    displayFlattenedFrame(diffScene, 100, magnification=10, differenceImage=True,\
#        filename="diff_scene.png")



if fourierApproach:
    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)

    flatFrameLength = len(listOfFlatFrames[0])

    listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
        for i in range(len(listOfFlatFrames) - 1)]

#    print listOfFlatDifferenceFrames[0].shape

    concatenatedDifferenceFrames = np.concatenate(listOfFlatDifferenceFrames, 1)

#    print concatenatedDifferenceFrames.shape

#    displayConcatenatedArray(concatenatedDifferenceFrames, magnification=10, \
#        differenceImage=True)

#    smallOccluderArray = np.array([0,0,1,0,1,0,0,0,0])
#    occluderArray = stretchOccluderArray(smallOccluderArray, 2*flatFrameLength-1)

    oldFlatFrame = listOfFlatFrames[0]
    flatFrameLength = len(oldFlatFrame)

    flatFrame2 = listOfFlatFrames[100]
    flatFrame1 = listOfFlatFrames[99]

    diffScene = flatFrame2 - flatFrame1

#    displayFlattenedFrame(diffScene, 100, magnification=10, differenceImage=True,\
#        filename="diff_scene.png")

    occluderArray = [0]*int(2*flatFrameLength) + \
        generateRandomOccluderList(int(2*flatFrameLength), 0.005)
    occluderArray = occluderArray + [0]*((6*flatFrameLength-1) - len(occluderArray))
    # Good lord that is gross


#    occluderArray = generateRandomOccluder(2*flatFrameLength-1, 0.005)


    transferMatrix = getTransferMatrixFromOccluder(occluderArray, flatFrameLength)

    distortedFlatFrame1 = np.dot(transferMatrix, diffScene)

#    displayFlattenedFrame(diffScene, 200,\
#        magnification=10, differenceImage=True)

    diffSceneFT = np.fft.fft(diffScene, axis=0)
    occluderArrayFT = np.fft.fft(occluderArray)
    distortedFlatFrameFT = np.fft.fft(distortedFlatFrame1, axis=0)

#    displayFlattenedFrame(diffSceneFT, 200,\
#        magnification=10, differenceImage=True)

    displayFlattenedFrame(blackAndWhiteifyFlatFrame(occluderArray), 200,\
        magnification=255, differenceImage=False)

#    displayFlattenedFrame(blackAndWhiteifyFlatFrame(occluderArrayFT), 200,\
#        magnification=255, differenceImage=False)

#    displayFlattenedFrame(distortedFlatFrameFT, 200,\
#        magnification=10, differenceImage=False)

#    displayFlattenedFrame(blackAndWhiteifyFlatFrame(occluderArray), 200, \
#        magnification=255, differenceImage=False)

#    displayFlattenedFrame(np.fft.ifft(np.divide(distortedFlatFrameFT, \
#        diffSceneFT)), 200, magnification=10, differenceImage=False)

#    displayFlattenedFrame(blackAndWhiteifyFlatFrame(np.fft.fft(occluderArray).real),
#        200, magnification=255, differenceImage=False)

#    displayFlattenedFrame(blackAndWhiteifyFlatFrame(np.fft.fft(occluderArray).imag),
#        200, magnification=255, differenceImage=False)

if getOutMovie:

    listOfFlatFrames = batchList(pickle.load(open("flat_frames_grey_bar_obs.p", "r")), 20)
#    listOfFlatFrames = batchList(pickle.load(open("real_flat_frames.p", "r")), 20)

    print listOfFlatFrames[0].shape

    flatFrameLength = len(listOfFlatFrames[0])

    print dVideoDT(listOfFlatFrames).shape

    diffObs = np.concatenate(dVideoDT(listOfFlatFrames), 1)

    print diffObs.shape

    occluderArray = stretchOccluderArray(REAL_OCCLUDER_ARRAY, 2*flatFrameLength-1)

    displayConcatenatedArray(diffObs, magnification=10, \
        differenceImage=True)

    transferMatrix = getTransferMatrixFromOccluder(occluderArray, \
        flatFrameLength)

    recoveredScene = recoverScene(transferMatrix, \
        diffObs, alpha=0.01)

    pickle.dump(recoveredScene, open("recov.p","w"))

    displayConcatenatedArray(np.transpose(recoveredScene), magnification=10, \
        differenceImage=True)

if readMovie:
    movie = pickle.load(open("recov.p", "r"))

    print movie.shape

    displayConcatenatedArray(movie, magnification=1000, differenceImage=True)


if findOccluderWithSparsityEntireHistory:
    FILE_NAME = "smaller_movie.mov"
    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)

    flatFrameLength = len(listOfFlatFrames[0])

    listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
        for i in range(len(listOfFlatFrames) - 1)]

#    print listOfFlatDifferenceFrames[0].shape

    concatenatedDifferenceFrames = np.concatenate(listOfFlatDifferenceFrames, 1)

#    print concatenatedDifferenceFrames.shape

    print concatenatedDifferenceFrames.shape

    displayConcatenatedArray(concatenatedDifferenceFrames, magnification=10, \
        differenceImage=True)

    smallOccluderArray = np.array([0,0,0,0,1,0,0,0,0])
    occluderArray = stretchOccluderArray(smallOccluderArray, 2*flatFrameLength-1)

    np.fft.fft(smallOccluderArray)

    otherSmallOccluderArray = np.array([0,0,0,0,1,0,0,0,0])
    otherRandomOccluder = stretchOccluderArray(otherSmallOccluderArray, 2*flatFrameLength-1)

    displayFlattenedFrame(blackAndWhiteifyFlatFrame(otherRandomOccluder), 200, magnification=255, differenceImage=False,\
        filename="occluder.png")

    transferMatrix = getTransferMatrixFromOccluder(occluderArray, flatFrameLength)
    otherTransferMatrix = getTransferMatrixFromOccluder(otherRandomOccluder, \
        flatFrameLength)

    print transferMatrix.shape

    diffObs = np.dot(transferMatrix, concatenatedDifferenceFrames)
    displayConcatenatedArray(diffObs, magnification=10, \
        differenceImage=True)

    recoveredSceneWithIncorrectOccluder = np.swapaxes(recoverScene(otherTransferMatrix, \
        diffObs, alpha=0.01), 0,1)

    print "done with inversion"

    displayConcatenatedArray(recoveredSceneWithIncorrectOccluder, magnification=10, \
        differenceImage=True)

    pickle.dump(recoveredSceneWithIncorrectOccluder, open("correct.p", "w"))

#    scoreOfCorrectOccluder = np.linalg.norm(np.dot(transferMatrix, \
#        recoveredSceneWithIncorrectOccluder) - diffObs) + \
#        0.01*np.linalg.norm(recoveredSceneWithIncorrectOccluder, 1)

#    scoreOfIncorrectOccluder = np.linalg.norm(np.dot(otherTransferMatrix, \
#        recoveredSceneWithCorrectOccluder) - diffObs) + \
#        0.01*np.linalg.norm(recoveredSceneWithCorrectOccluder, 1)

    print "score of correct",

if findOccluderWithSparsity:
    FILE_NAME = "smaller_movie.mov"
    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)

    oldFlatFrame = listOfFlatFrames[0]
    flatFrameLength = len(oldFlatFrame)

    flatFrame2 = listOfFlatFrames[100]
    flatFrame1 = listOfFlatFrames[99]

    diffScene = flatFrame2 - flatFrame1

    displayFlattenedFrame(diffScene, 100, magnification=10, differenceImage=True,\
        filename="diff_scene.png")

    occluderArray = generateRandomOccluder(2*flatFrameLength-1, 0.005)

#    smallOccluderArray = generateRandomOccluder(9, 0.5)
    smallOccluderArray = np.array([0,0,0,0,1,0,0,0,0])
    occluderArray = stretchOccluderArray(smallOccluderArray, 2*flatFrameLength-1)

    otherSmallOccluderArray = np.array([0,0,0,0,1,0,1,0,0])
    otherRandomOccluder = stretchOccluderArray(otherSmallOccluderArray, 2*flatFrameLength-1)

    transferMatrix = getTransferMatrixFromOccluder(occluderArray, flatFrameLength)

    displayFlattenedFrame(blackAndWhiteifyFlatFrame(occluderArray), 200, magnification=255, differenceImage=False,\
        filename="occluder.png")

    diffObs = np.dot(transferMatrix, diffScene)

    displayFlattenedFrame(diffObs, 100, magnification=1, differenceImage=True,\
        filename="diff_obs.png")

#    otherRandomOccluder = generateRandomOccluder(2*flatFrameLength-1, 0.005)

    displayFlattenedFrame(blackAndWhiteifyFlatFrame(otherRandomOccluder), 200, magnification=255, differenceImage=False,\
        filename="other_occluder.png")

    otherTransferMatrix = getTransferMatrixFromOccluder(otherRandomOccluder, \
        flatFrameLength)

    recoveredSceneWithCorrectOccluder = np.swapaxes(recoverScene(transferMatrix, \
        diffObs, alpha=0.01), 0,1)

    displayFlattenedFrame(recoveredSceneWithCorrectOccluder, 100, magnification=10,\
        differenceImage=True, filename="recovered_scene_correct.png")

    scoreOfCorrectOccluder = np.linalg.norm(np.dot(transferMatrix, \
        recoveredSceneWithCorrectOccluder) - diffObs) + \
        0.01*np.linalg.norm(recoveredSceneWithCorrectOccluder, 1)

    print scoreOfCorrectOccluder

    recoveredSceneWithIncorrectOccluder = np.swapaxes(recoverScene(otherTransferMatrix, \
        diffObs, alpha=0.01), 0,1)
    scoreOfIncorrectOccluder = np.linalg.norm(np.dot(otherTransferMatrix, \
        recoveredSceneWithIncorrectOccluder) - diffObs) + \
        0.01*np.linalg.norm(recoveredSceneWithIncorrectOccluder, 1)

#    print scoreOfIncorrectOccluder



    recoveredScene = recoverScene(otherTransferMatrix, diffObs, alpha=0.01)
    displayFlattenedFrame(np.swapaxes(recoveredScene,0,1), 100, magnification=10,\
        differenceImage=True, filename="recovered_scene_incorrect.png")

if exportFlatVideo:
#    trueOccluder = generateRandomOccluder(9)

#    FILE_NAME = "smaller_movie.mov"
    FILE_NAME = "grey_bar_movie.mpeg"
    vid = imageio.get_reader(FILE_NAME,  'ffmpeg')

    listOfFlattenedFrames = turnVideoIntoListOfFlattenedFrames(vid)

    flatFrameLength = len(listOfFlattenedFrames[0])

    occluderArray = stretchOccluderArray(REAL_OCCLUDER_ARRAY, 2*flatFrameLength-1)

    transferMatrix = getTransferMatrixFromOccluderLong(occluderArray, flatFrameLength)

#    firstFlatFrame = listOfFlatFrames[0]
#    flatFrameLength = len(oldFlatFrame)

    listOfFlatFramesObs = [np.dot(transferMatrix, frame) for frame in \
        listOfFlattenedFrames]

    pickle.dump(listOfFlattenedFrames, open("flat_frames_grey_bar.p", "w"))

    pickle.dump(listOfFlatFramesObs, open("flat_frames_grey_bar_obs.p", "w"))


#listOfFrames = turnVideoIntoListOfFrames(vid)

if displayFrameOfVideo:
    FILE_NAME = "smaller_movie.mov"

    vid = imageio.get_reader(FILE_NAME,  'ffmpeg')

    frameNum1 = 200
    im = vid.get_data(frameNum1)
    frame1 = np.array(im).astype(float)

    frameNum2 = frameNum1 + 1
    im = vid.get_data(frameNum2)
    frame2 = np.array(im).astype(float)

    viewFrame(frame2 - frame1, magnification=10, differenceImage=True)

if displayFlatFrameOfVideo:
    FILE_NAME = "smaller_movie.mov"
    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)

    oldFlatFrame = listOfFlatFrames[0]
    flatFrameLength = len(oldFlatFrame)

    flatFrame2 = listOfFlatFrames[100]
    flatFrame1 = listOfFlatFrames[99]

    diffScene = flatFrame2 - flatFrame1

#    displayFlattenedFrame(diffScene, 100, magnification=10, differenceImage=True,\
#        filename="diff_scene.png")

    occluderArray = generateRandomOccluder(2*flatFrameLength-1, 0.005)
    transferMatrix = getTransferMatrixFromOccluder(occluderArray, flatFrameLength)

    distortedFlatFrame1 = np.dot(transferMatrix, flatFrame1)
    displayFlattenedFrame(blackAndWhiteifyFlatFrame(occluderArray), 1000, magnification=255,  differenceImage=False, \
        filename="occluder.png")

    displayFlattenedFrame(distortedFlatFrame1, 1000, magnification=1e-3, differenceImage=False,\
            filename="distorted_frame.png")

if buildVideos:

    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)
#    listOfFlatFrames = batchList(pickle.load(open("flat_frames_grey_bar.p", "r")), 4)
#    listOfFlatFrames = batchList(pickle.load(open("real_flat_frames.p", "r")), 20)


    oldFlatFrame = listOfFlatFrames[0]
    flatFrameLength = len(oldFlatFrame)

#    occluderArray = generateRandomOccluder(2*flatFrameLength-1, 0.005)
    occluderArray = np.array(REAL_OCCLUDER_ARRAY)
#    smallOccluderArray = generateRandomOccluder(9, 0.5)
#    smallOccluderArray = np.array([0,0,0,0,1,0,0,0,0])
#    occluderArray = stretchOccluderArray(smallOccluderArray, 2*flatFrameLength-1)

    print len(occluderArray), 2*flatFrameLength-1

#    transferMatrix = getTransferMatrixFromOccluderLong(occluderArray, flatFrameLength)
    transferMatrix = differentDepthsTransferMatrix(flatFrameLength)

    print "shape", transferMatrix.shape
#    print "det", np.linalg.det(transferMatrix)

    displayFlattenedFrame(blackAndWhiteifyFlatFrame(occluderArray), 200, magnification=255,
        differenceImage=False, filename="occluder.png")

    frameCounter = 0

    sceneMovie = []
    differenceSceneMovie = []
    differenceObsMovie = []
    obsMovie = []

    ROWS_PER_FRAME = 10

    while frameCounter < len(listOfFlatFrames):
        print frameCounter
        newFlatFrame = listOfFlatFrames[frameCounter]
        differenceFrame = newFlatFrame - oldFlatFrame
        obsFrame = np.dot(transferMatrix, newFlatFrame)

    #    p.matshow(transferMatrix)
    #    p.show()
        garbledDifferenceFrame = np.dot(transferMatrix, differenceFrame)
    #    print garbledDifferenceFrame
#        displayFlattenedFrame(newFlatFrame, height=100, magnification=1, \
#            differenceImage=False, filename="flat_frames/flat_frame_truth_" + \
#            str(frameCounter) + ".png")
#        displayFlattenedFrame(differenceFrame, height=100, magnification=10, \
#            differenceImage=True, filename="flat_frames/flat_frame_diff_" + str(frameCounter) + \
#            ".png")
#        displayFlattenedFrame(garbledDifferenceFrame, 100, 1, True, \
#            filename="flat_frames/flat_frame_conv_" + str(frameCounter) + \
#            ".png")



        for _ in range(ROWS_PER_FRAME):
            sceneMovie.append(newFlatFrame)
            differenceSceneMovie.append(differenceFrame)
            obsMovie.append(obsFrame)
#            print garbledDifferenceFrame, np.average(garbledDifferenceFrame)

            differenceObsMovie.append(garbledDifferenceFrame - \
                [average(garbledDifferenceFrame)]*len(garbledDifferenceFrame))

        oldFlatFrame = newFlatFrame

        frameCounter += 1

#    recoveredScene = recoverScene(transferMatrix, np.array(differenceSceneMovie), alpha=0.01)

#    viewFrame(recoveredScene, differenceImage=False, magnification=1, \
#        filename="recovered_scene_movie.png")


    viewFrame(np.array(sceneMovie), differenceImage=False, magnification=1, \
        filename="scene_movie.png")
    viewFrame(np.array(differenceSceneMovie), differenceImage=True, magnification=10, \
        filename="diff_scene_movie.png")
    viewFrame(np.array(obsMovie), differenceImage=False, magnification=1e-3, \
        filename="obs_movie.png")
    viewFrame(np.array(differenceObsMovie), differenceImage=True, magnification=1, \
        filename="diff_obs_movie.png")

    #imRaw = Image.open("adam_h.jpeg").convert("RGB")
    #frame = np.array(imRaw).astype(float)

    #print displayFlattenedFrame(flattenFrame(frame), 100)

if exportFlatRealVideo:
    MOVIE_NAME = "grey_bar_obs_movie_real.MOV"
    vid = imageio.get_reader(MOVIE_NAME)

    print len(vid)

    im = vid.get_data(0)
    frame = np.array(im).astype(float)

    frameShape = frame.shape
    frameHeight = frameShape[0]

#    viewFrame(frame)


    LEFT_BOTTOM_PIXEL = 61
    RIGHT_BOTTOM_PIXEL = 1179

#    flatFrame = turnRealFrameIntoFlatFrame(frame, LEFT_BOTTOM_PIXEL, \
#        RIGHT_BOTTOM_PIXEL)

    listOfFlatFrames = []

    for i in range(2500, 4500):
        print i-2500, "/", 2000
        im = vid.get_data(i)
        frame = np.array(im).astype(float)
        flatFrame = turnRealFrameIntoFlatFrame(frame, LEFT_BOTTOM_PIXEL, \
                RIGHT_BOTTOM_PIXEL)

        listOfFlatFrames.append(flatFrame)

    pickle.dump(listOfFlatFrames, open("real_flat_frames_2500_4500.p", "w"))

#    displayFlattenedFrame(flatFrame, 200)

if showArtificialOccluder:
    displayFlattenedFrame(blackAndWhiteifyFlatFrame(REAL_OCCLUDER_ARRAY), 200,\
        magnification=255, differenceImage=False)

    transferMatrix = getTransferMatrixFromOccluder(REAL_OCCLUDER_ARRAY, flatFrameLength)

    scoreOfCorrectOccluder = np.linalg.norm(np.dot(transferMatrix, \
            recoveredSceneWithIncorrectOccluder) - diffObs) + \
            0.01*np.linalg.norm(recoveredSceneWithIncorrectOccluder, 1)

if testDifferentDepthsOccluder:
    n = 100
#    p.matshow(verticalTransferMatrix(n))
    p.matshow(np.minimum(verticalTransferMatrix(n),horizontalTransferMatrix(n)))
    p.show()

if displaySimpleOccluder:
#    smallOccluderArray = np.array([0,0,0,0,1,0,0,0,0])
    smallOccluderArray = REAL_OCCLUDER_ARRAY

    displayFlattenedFrame(blackAndWhiteifyFlatFrame(smallOccluderArray), 300, magnification=255)

if polynomialApproach:
    FILE_NAME = "smaller_movie.mov"
    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)

    flatFrameLength = len(listOfFlatFrames[0])

    listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
        for i in range(len(listOfFlatFrames) - 1)]

#    print listOfFlatDifferenceFrames[0].shape

    concatenatedDifferenceFrames = np.concatenate(listOfFlatDifferenceFrames, 1)

#    print concatenatedDifferenceFrames.shape

    print concatenatedDifferenceFrames.shape

    displayConcatenatedArray(concatenatedDifferenceFrames, magnification=10, \
        differenceImage=True)
        
    smallOccluderArray = np.array([0,0,0,0,1,0,0,0,0])
    occluderArray = stretchOccluderArray(smallOccluderArray, 2*flatFrameLength-1)

    np.fft.fft(smallOccluderArray)

    otherSmallOccluderArray = np.array([0,0,0,0,1,0,0,0,0])
    otherRandomOccluder = stretchOccluderArray(otherSmallOccluderArray, 2*flatFrameLength-1)

    displayFlattenedFrame(blackAndWhiteifyFlatFrame(otherRandomOccluder), 200, magnification=255, 
        differenceImage=False, filename="occluder.png")

    transferMatrix = getTransferMatrixFromOccluder(occluderArray, flatFrameLength)
    otherTransferMatrix = getTransferMatrixFromOccluder(otherRandomOccluder, \
        flatFrameLength)

    print transferMatrix.shape

    diffObs = np.dot(transferMatrix, concatenatedDifferenceFrames)
    displayConcatenatedArray(diffObs, magnification=10, \
        differenceImage=True)

    recoveredSceneWithIncorrectOccluder = np.swapaxes(recoverScene(otherTransferMatrix, \
        diffObs, alpha=0.01), 0,1)

    print "done with inversion"

    displayConcatenatedArray(recoveredSceneWithIncorrectOccluder, magnification=10, \
        differenceImage=True)
    