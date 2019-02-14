from __future__ import division
import numpy as np
import random
from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
from image_distortion_simulator import imageify, imageifyComplex
from scipy.signal import convolve, deconvolve, argrelextrema, convolve2d, correlate2d, medfilt
from numpy.fft import fft, ifft
from math import sqrt, pi, exp, log, sin, cos, floor, ceil
from import_1dify import batchList, displayConcatenatedArray, fuzzyLookup, stretchArray, fuzzyLookup2D
from image_distortion_simulator import doFuncToEachChannelVec, doFuncToEachChannel, circleSpeck,\
    getAttenuationMatrixOneOverF, getAttenuationMatrix, swapChannels, doFuncToEachChannelSeparated, \
    doSeparateFuncToEachChannel, doSeparateFuncToEachChannelSeparated, doFuncToEachChannelSeparatedTwoInputs, \
    doFuncToEachChannelTwoInputs
import pickle
from numpy.polynomial.polynomial import Polynomial, polyfromroots
import matplotlib.pyplot as p
from scipy.linalg import circulant as circ
from scipy.special import gamma
import matplotlib.cm as cm 
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator
from matplotlib import rcParams
from custom_plot import createCMapDictHelix, LogLaplaceScale
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import broyden1, fmin_bfgs, fmin_cg, fmin_tnc, fmin
from pynverse import inversefunc
from scipy.integrate import quad
from video_processor import batchArrayAlongAxis, batchAndDifferentiate, convertArrayToVideo
from scipy.linalg import toeplitz
from scipy.linalg import dft as dftMat
from phase_retrieval import retrievePhase, retrievePhase2D
import imageio
from best_matrix import padIntegerWithZeros
from scipy.ndimage import gaussian_filter
from PIL import Image
from skimage import color, data, restoration
import scipy.io

LOOK_AT_FREQUENCY_PROFILES = False
DIVIDE_OUT_STRATEGY = False
POLYNOMIAL_STRATEGY = False
UNDO_TRUNCATION = False
BILL_NOISY_POLYNOMIALS = False
DENSITY_TESTS = False
GIANT_GRADIENT_DESCENT = False
ROADSIDE_DECONVOLUTION = False
POLYNOMIAL_EXPERIMENT = False
POLYNOMIAL_EXPERIMENT_AFTERMATH = False
POLYNOMIAL_EXPERIMENT_FRANKENSTEIN = False
POLYNOMIAL_EXPERIMENT_RECOVERY = False
SIMPLE_BLIND_DECONV_TEST = False
RECOVER_FREQ_MAGNITUDES = False
RECOVER_FREQ_MAGNITUDES_2 = False
BLIND_2D_SIM = False
STUPID_METHOD_2D = False
SPLIT_FRAMES = False
FOURIER_BURST_ACCUMULATION = False
LOOK_AT_AGG_FRAME = False
STUPID_METHOD_2D_2 = False
STUPID_METHOD_2D_3 = False
STUPID_METHOD_2D_4 = False
ANALYZE_EXTRACTED_OCCLUDER = False
CREATE_RECONSTRUCTION_MOVIE = False
VIEW_FRAMES = False
DIFF_EXP_VIDEO = False
CONV_SIM_VIDEO = False
PROCESS_SIM_VIDEO = True
PROCESS_EXP_VIDEO = False
VIEW_OCC = False
OVERLAP_PAD_TEST = False
OVERLAP_PAD_TEST_2 = False
PROCESS_EXP_OCCLUDER = False
PROCESS_EXP_OCCLUDER_2 = False
PROCESS_EXP_OCCLUDER_BIN_ONLY = False
PROCESS_EXP_OCCLUDER_CARDBOARD = False
CREATE_RECONSTRUCTION_MOVIE_EXP = False
CREATE_RECONSTRUCTION_MOVIE_EXP_2 = False
EXTRACT_MATRIX_FROM_IMAGE = False
WIENER_FILTER_TEST = False
CROP = False
DOWNSAMPLE_VID = False
MEAN_SUBTRACTION = False
MEAN_SUBTRACTION_POSITIVE = False
UNIFORM_MEAN_SUBTRACTION = False
SIM_COMPARISON = False
TIME_DIFF = False
MED_FILT = False
AVERAGE_DIVIDE = False
AVERAGE_DIVIDE_1D = False
GET_ABS = False
COLOR_AVG = False
COLOR_FLATTEN = False
MAKE_VIDEO = False
CAP_ARR_VALS = False
DOWNSIZE_ARR = False
MACARENA_CORRECT_DECONV_DUMPER = False
MACARENA_EXP_DECONV_DUMPER = False
EXP_63225_DECONV_DUMPER = False
ORANGE_DECONV_DUMPER = False
MACARENA_TEST = False
MACARENA_CORRECT_DECONV = False
MACARENA_EXP_DECONV = False
BLD66_DECONV_DUMPER = False
RECOVER_SCENE = False
CONVERT_ARRAYS_CHRISTOS = False
MESS_AROUND_WITH_OBS = False
MESS_AROUND_WITH_OBS_2 = False
AUTOCORR_TEST = False
AUTOCORR_TEST_2 = False
JANK_RECOVERY = False
CVPR_EXAMPLES = False
CVPR_COMPUTE_HAMMING_DISTANCE = False
CVPR_MAKE_EXAMPLE_MOVIE = False
CVPR_MAKE_EXAMPLE_MOVIE_2 = False

ZERO_DENSITY = 2
NONZERO_DENSITY = 20
FLIP_DENSITY = 10
SIGNAL_SIGMA = 1
NOISE_SIGMA = 1000

output = open("trash.txt", "w")

def pront(x):
    print x

def divideNoDivZero(x, y):
    if y == 0:
        if x == 0:
            return 0
        elif x > 0:
            return 10
        elif x < 0:
            return -10
    else:
        return x / y


numpyDivideNoDivZero = np.vectorize(divideNoDivZero)
mscale.register_scale(LogLaplaceScale)

def generateSparseFrame(x, y):
    protoOnOrOff = np.random.binomial(1, 0.3, (x, y))
    onOrOff = majorityGame(protoOnOrOff, 10)

    shotVal = np.random.exponential(1, (x, y))

    return np.multiply(onOrOff, shotVal)

def majorityGame(arr, coolingTime):
    maxX = arr.shape[0]
    maxY = arr.shape[1]

    for i in range(coolingTime):
        newArr = np.zeros(arr.shape)

        print i

        for x in range(-1, maxX-1):
            for y in range(-1, maxY-1):
                voteCounter = 0
                for deltaX in [-1, 0, 1]:
                    for deltaY in [-1, 0, 1]:
                        voteCounter += arr[x+deltaX][y+deltaY]
                if voteCounter > 4:
                    newArr[x][y] = 1

        arr = newArr
    return arr

def extractMatrixFromBWImage(imPath, pixelsPerPixel):
    imRaw = Image.open(imPath)
    arr = np.array(imRaw).astype(float)
    arrShape = arr.shape

    processedArr = []

    prevVal = float("Inf")

    i = pixelsPerPixel/2

    while i < arrShape[0]:
        j = pixelsPerPixel/2
        processedArr.append([])

        while j < arrShape[1]:
            processedArr[-1].append(arr[int(i)][int(j)][0])

            j += pixelsPerPixel

        i += pixelsPerPixel

    return np.array(processedArr)

def hammingDistance(arr1, arr2):
    return np.sum(np.abs(arr1-arr2))

def sparsify(arr, cutoffFactor=1):
    averageVal = np.sum(np.abs(arr))/np.size(arr)

    biggerThanCutoff = np.vectorize(lambda x: 1*(x>averageVal/cutoffFactor))

    viewFrame(imageify(biggerThanCutoff(np.abs(arr))), adaptiveScaling=True, \
        differenceImage=True)

    return np.multiply(arr, biggerThanCutoff(np.abs(arr)))

def largeMajorityGame(arr, coolingTime, voteRadius):
    maxX = arr.shape[0]
    maxY = arr.shape[1]

    for i in range(coolingTime):
        newArr = np.zeros(arr.shape)

        print i

        for x in range(-1, maxX-1):
            for y in range(-1, maxY-1):
                voteCounter = 0
                numVotes = 0
                for deltaX in range(-int(ceil(voteRadius)), int(ceil(voteRadius))+1):
                    for deltaY in range(-int(ceil(voteRadius)), int(ceil(voteRadius))+1):
                        if sqrt(deltaX*deltaX + deltaY*deltaY) < voteRadius:
                            if x+deltaX < maxX and x-deltaX >= 0:
                                if y+deltaY < maxY and y-deltaY >= 0: 
                                    voteCounter += arr[x+deltaX][y+deltaY]
                                    numVotes += 1

                if voteCounter/numVotes > 1/2:
                    newArr[x][y] = 1

        arr = newArr
    return arr    

def generateSparseMovie(x, y, t):
    return [generateSparseFrame(x, y) for _ in range(t)]

def generateRandomCorrelatedOccluder(x, y):
    arr = np.random.binomial(1, 0.5, (x, y))

    return majorityGame(arr, 10)

def generateRandomVeryCorrelatedOccluder(x, y, voteRadius):
    arr = np.random.binomial(1, 0.5, (x, y))

    return largeMajorityGame(arr, 6, voteRadius)

def convolve2DToeplitz(arr1, arr2):
    return convolve2d(arr1, arr2, mode="valid")

def convolve2DToeplitzFull(arr1, arr2):
    return convolve2d(arr1, arr2, mode="full")

def convolve2D(arr1, arr2):
    return np.real(np.fft.ifft2(np.multiply(np.fft.fft2(arr1), np.fft.fft2(arr2))))

def deconvolve2D(arr1, arr2):
    return np.real(np.fft.ifft2(np.multiply(np.fft.fft2(arr1), np.fft.fft2(arr2))))

def convolve2Images(arr1, arr2):
    return convolve2d(arr1, arr2, mode="full", boundary="fill", fillvalue=0)    

def makeImpulseFrame(shape, impulseLoc):
    returnArray = np.zeros(shape)
    returnArray[impulseLoc] += 1
    return returnArray

def makeRandomImpulseFrame(shape):
    returnArray = np.zeros(shape)
    impulseLoc = tuple([random.randint(0,i-1) for i in shape])

    returnArray[impulseLoc] += 1
    return returnArray

def getGaussianKernel(squareDiameter, pixelsPerSigma):
    centerPoint = (squareDiameter-1)/2

    returnArray = []

    for i in range(squareDiameter):
        returnArray.append([])
        for j in range(squareDiameter):
            returnArray[-1].append(exp(-(((i-centerPoint)/pixelsPerSigma)**2 + \
                                        ((j-centerPoint)/pixelsPerSigma)**2)))

    return np.array(returnArray)

def getGaussianKernelVariableLocation(arrShape, centerPoint, pixelsPerSigma):

    returnArray = []

    for i in range(arrShape[0]):
        returnArray.append([])
        for j in range(arrShape[1]):
            returnArray[-1].append(exp(-(((i-centerPoint[0])/pixelsPerSigma)**2 + \
                                        ((j-centerPoint[1])/pixelsPerSigma)**2)))

    return np.array(returnArray)    

def makeMultipleImpulseFrame(shape, numImpulses):
    returnArray = np.zeros(shape)
    impulseLocs = [tuple([random.randint(0,i-1) for i in shape]) for _ in range(numImpulses)]

    for impulseLoc in impulseLocs:
        returnArray[impulseLoc] += 1

    return returnArray

def multidim_cumsum(a):
    out = a[...,::-1].cumsum(-1)[...,::-1]
    for i in range(2,a.ndim+1):
        np.cumsum(out, axis=-i, out=out)
    return out

def getCompensationFactorFromArray(a):
    upperRight = multidim_cumsum(a)
    upperLeft = np.flip(multidim_cumsum(np.flip(a, 0)), 0)
    lowerRight = np.flip(multidim_cumsum(np.flip(a, 1)), 1)
    lowerLeft = np.flip(np.flip(multidim_cumsum(np.flip(np.flip(a, 0), 1)), 0), 1)

    overall = np.concatenate([np.concatenate([lowerRight, lowerLeft], 0),\
                np.concatenate([upperRight, upperLeft], 0)], 1)

    overallShape = overall.shape

    overall = np.delete(np.delete(overall, int(overallShape[0]/2), 0), int(overallShape[1]/2), 1)

#    print overall.shape
#    viewFrame(imageify(overall), adaptiveScaling=True)

    return overall


def getCenteringFactor(i, j, arrShape0, arrShape1, minCompensation):
    xCompensation = arrShape0 - abs(i - arrShape0)
    yCompensation = arrShape1 - abs(j - arrShape1)

    return np.power(max(xCompensation*yCompensation, minCompensation*xCompensation, minCompensation*yCompensation, \
        minCompensation**2), 0.2)

def getCenteringArray(arrShape):

    ciShape = [2*i-1 for i in arrShape]

    return np.transpose(np.array([[getCenteringFactor(i,j,arrShape[0], \
            arrShape[1],0) for i in \
            range(ciShape[0])] for j in range(ciShape[1])])) 


def getCompensationArray(arr1, arr2, minCompensation):
#    print arr1.shape
#    print arr2.shape

    ca1 = getCompensationFactorFromArray(arr1)
    ca2 = getCompensationFactorFromArray(arr2)

#    print ca1.shape
#    print ca2.shape

    return np.maximum(np.multiply(np.sqrt(ca1), doubleFlip(np.sqrt(ca2))), minCompensation*np.ones(ca1.shape))

def doubleFlip(a):
    return np.flip(np.flip(a, 0), 1)

def getMatchArrayUnequalSize(arr1, arr2, minCompensation=10):
    print "hi"
    matchArray = convolve2DToeplitzFull(arr1, doubleFlip(arr2))

    viewFrame(imageify(matchArray), adaptiveScaling=True)

    bestIndex = np.unravel_index(np.argmax(matchArray, axis=None), matchArray.shape)    

    print bestIndex

    return matchArray, bestIndex

def getMatchArray(arr1, arr2, minCompensation=10):
    arrShape = arr1.shape

#    centeringArray = getCenteringArray(arrShape)

#    compensationArray = getCompensationArray(np.abs(arr1), np.abs(arr2), minCompensation)

    convolvedArray = np.abs(convolve2d(arr1, doubleFlip(arr2)))

#    matchArray = np.multiply(np.divide(convolvedArray, compensationArray), centeringArray)
    
    matchArray = convolvedArray

    bestIndex = np.unravel_index(np.argmax(matchArray, axis=None), matchArray.shape)

    matchQuality = matchArray[bestIndex]

    bestMatchArray = np.zeros(matchArray.shape)
    bestMatchArray[bestIndex] += 1



    return matchArray, bestMatchArray, bestIndex, matchQuality

def getMatchArrayRGB(arr1, arr2, minCompensation=10):
    arrDims = arr1.shape[:-1]

    convolvedArrays = doFuncToEachChannelSeparatedTwoInputs(convolve2d, arr1, doubleFlip(arr2))

    matchArray = np.abs(convolvedArrays[0]) + np.abs(convolvedArrays[1]) + np.abs(convolvedArrays[2])

    bestIndex = np.unravel_index(np.argmax(matchArray, axis=None), matchArray.shape)

    return matchArray, bestIndex

#    viewFrame(imageify(matchArray), adaptiveScaling=True)

def getMatchArrayRGBNegative(arr1, arr2, minCompensation=10):
    arrDims = arr1.shape[:-1]

    convolvedArrays = doFuncToEachChannelSeparatedTwoInputs(convolve2d, arr1, doubleFlip(arr2))

    matchArray = convolvedArrays[0] + convolvedArrays[1] + convolvedArrays[2]

    return matchArray

def getMatchArrayImage(arr1, arr2, minCompensation=10):

    arrDims = arr1.shape[:-1]

    matchArray = doFuncToEachChannelTwoInputs(convolve2d, arr1, doubleFlip(arr2))

    return matchArray
#    viewFrame(imageify(matchArray), adaptiveScaling=True)


def getOverlapArray(arr1, arr2, bestIndex):
    arrShape = arr1.shape

    firstIndex = bestIndex[0] - arrShape[0]
    secondIndex = bestIndex[1] - arrShape[1]

#    print "first, second", firstIndex, secondIndex

    if firstIndex >= 1 and secondIndex >= 1:
        arr2Snapshot = arr2[:-firstIndex+1,:-secondIndex+1]
        arr1Snapshot = arr1[firstIndex-1:,secondIndex-1:]

    elif firstIndex < 1 and secondIndex >= 1:
        arr2Snapshot = arr2[-firstIndex+1:,:-secondIndex+1]
        arr1Snapshot = arr1[:firstIndex-1,secondIndex-1:]        

    elif firstIndex >= 1 and secondIndex < 1:
        arr2Snapshot = arr2[:-firstIndex+1,-secondIndex+1:]
        arr1Snapshot = arr1[firstIndex-1:,:secondIndex-1]

    elif firstIndex < 1 and secondIndex < 1:
        arr2Snapshot = arr2[-firstIndex+1:,-secondIndex+1:]
        arr1Snapshot = arr1[:firstIndex-1,:secondIndex-1]        

    return arr1Snapshot, arr2Snapshot

def getOverlapArrayFullCanvas(canvas, otherArray, bestIndex):
    arrShape = canvas.shape

    firstIndex = bestIndex[0] - arrShape[0]
    secondIndex = bestIndex[1] - arrShape[1]

    print "first, second", firstIndex, secondIndex

    if firstIndex >= 0 and secondIndex >= 0:
        otherArraySnapshot = np.pad(otherArray[:-firstIndex-1,:-secondIndex-1], \
            [(firstIndex+1, 0), (secondIndex+1, 0)], "constant")
#        arr1Snapshot = arr1[firstIndex-1:,secondIndex-1:]

    elif firstIndex < 0 and secondIndex >= 0:
        otherArraySnapshot = np.pad(otherArray[-firstIndex-1:,:-secondIndex-1], \
            [(0, -firstIndex-1), (secondIndex+1, 0)], "constant")

    elif firstIndex >= 0 and secondIndex < 0:
        otherArraySnapshot = np.pad(otherArray[:-firstIndex-1,-secondIndex-1:], \
            [(firstIndex+1, 0), (0, -secondIndex-1)], "constant")

    elif firstIndex < 0 and secondIndex < 0:
        otherArraySnapshot = np.pad(otherArray[-firstIndex-1:,-secondIndex-1:], \
            [(0, -firstIndex-1), (0, -secondIndex-1)], "constant")

    return otherArraySnapshot

# doesn't assume arr1 and arr2 are the same shape
def getOverlapArrayPadded(arr1, arr2, bestIndex):
    arr1Shape = arr1.shape
    arr2Shape = arr2.shape

    firstIndex = bestIndex[0] - arrShape[0]
    secondIndex = bestIndex[1] - arrShape[1]    

    print firstIndex, secondIndex

    if firstIndex >= 0 and secondIndex >= 0:
        arr2Snapshot = np.pad(arr2, ((firstIndex+1, 0), (secondIndex+1, 0)), "constant")
        arr1Snapshot = np.pad(arr1, ((0, firstIndex+1), (0, secondIndex+1)), "constant")

    print arr2Snapshot.shape
    print arr1Snapshot.shape

    return arr1Snapshot, arr2Snapshot

def getOverlapArrayPadded(arr1, arr2, bestIndex):
    arr1Shape = arr1.shape
    arr2Shape = arr2.shape

    outShape = [i+j for i,j in zip(arr1Shape, arr2Shape)]


    firstIndex = bestIndex[0] - (arr1Shape[0] + arr2Shape[0])/2
    secondIndex = bestIndex[1] - (arr1Shape[1] + arr2Shape[1])/2   

    arr1Snapshot = np.pad(arr1, ((outShape[0]-bestIndex[0]+int(arr2Shape[0]/2), 
        bestIndex[0]+arr2Shape[0]-int(arr2Shape[0]/2)+1), \
        (outShape[1]-bestIndex[1]+int(arr2Shape[1]/2)+1, \
        bestIndex[1]+arr2Shape[1]-int(arr2Shape[1]/2))), "constant")
    arr2Snapshot = np.pad(arr2, ((outShape[0]-bestIndex[0]+int(arr1Shape[0]/2)+1, 
        bestIndex[0]+arr1Shape[0]-int(arr1Shape[0]/2)), \
        (outShape[1]-bestIndex[1]+int(arr1Shape[1]/2), \
        bestIndex[1]+arr1Shape[1]-int(arr1Shape[1]/2)+1)), "constant")

    return arr1Snapshot, arr2Snapshot

def isSparse(x):
    if abs(x) > 1e-4:
        return 0
    return 1

def sparsity(arr):
    return np.sum(np.sum(np.vectorize(isSparse)(arr)))

def getMags(arr):
    return np.real(np.fft.fft2(arr))

def generateSparseSeq(n):
    oddsOfZeroNonzeroTransition = ZERO_DENSITY/n

    oddsOfNonzeroZeroTransition = NONZERO_DENSITY/n

    returnList = []

    if random.random() < ZERO_DENSITY / (ZERO_DENSITY + NONZERO_DENSITY):
        currentState = "zero"

    else:
        currentState = "nonzero"


    for _ in range(n):
        if currentState == "zero":
            if random.random() < oddsOfZeroNonzeroTransition:
                currentState = "nonzero"

        elif currentState == "nonzero":
            if random.random() < oddsOfNonzeroZeroTransition:
                currentState = "zero"

        else:
            raise

        if currentState == "zero":
            returnList.append(0)

        elif currentState == "nonzero":
            returnList.append(np.random.normal(SIGNAL_SIGMA))

        else:
            raise

    return np.array(returnList)

def chargeOccluderElement(elt, lambda1):
    if elt >= 0 and elt <= 1:
        return 0

    else:
        return lambda1*min(abs(elt), abs(elt-1))

def occGradSparsitySingleElt(elt, lambda1):
    if elt >= 0 and elt <= 1:
        return 0        

    elif elt > 1:
        return lambda1

    elif elt < 0:
        return -lambda1

    else:
        raise

def occGradSparsity(occVal, lambda1):
    return np.array([occGradSparsitySingleElt(i, lambda1) for i in occVal])         

def chargeOccluderElementPair(elt1, elt2, lambda2):
    return lambda2*abs(elt1-elt2)

def occGradSpatialDoubleElt(occVal, otherOccVal, lambda2):
    if occVal == otherOccVal:
        return 0

    elif occVal > otherOccVal:
        return lambda2

    elif occVal < otherOccVal:
        return -lambda2

def occGradSpatialTripleElt(occValLeft, occValCenter, occValRight, lambda2):
    if occValLeft is None:
        return occGradSpatialDoubleElt(occValCenter, occValRight, lambda2)

    if occValRight is None:
        return occGradSpatialDoubleElt(occValCenter, occValLeft, lambda2)

    else:
        return occGradSpatialDoubleElt(occValCenter, occValRight, lambda2) + \
            occGradSpatialDoubleElt(occValCenter, occValLeft, lambda2)

def occGradSpatial(occVal, lambda2):
    return np.array([occGradSpatialTripleElt(None, occVal[0], occVal[1], lambda2)] + \
        [occGradSpatialTripleElt(occVal[i-1], occVal[i], occVal[i+1], lambda2) for i in range(1, len(occVal)-1)] + \
        [occGradSpatialTripleElt(occVal[-2], occVal[-1], None, lambda2)])

def chargeOccluder(occluderSeq, lambda1, lambda2):
    return sum([chargeOccluderElement(elt, lambda1) for elt in occluderSeq]) + \
        sum([chargeOccluderElementPair(occluderSeq[i], occluderSeq[i+1], lambda2) for i in range(len(occluderSeq)-1)])

def splitMovieIntoListOfFrames(movieSeq, frameLength):
    return np.reshape(movieSeq, (int(len(movieSeq)/frameLength), frameLength))

def chargeFramePair(frame1, frame2, lambda3):
    return lambda3*sum(np.abs(frame1-frame2))

def chargeFrameElement(elt, lambda4):
    return lambda4*abs(elt)

def chargeFrameElementPair(elt1, elt2, lambda5):
    return lambda5*abs(elt1-elt2)

def chargeFrame(frame, lambda4, lambda5):
    return sum([chargeFrameElement(elt, lambda4) for elt in frame]) + \
        sum([chargeFrameElementPair(frame[i], frame[i + 1], lambda5) for i in
            range(len(frame) - 1)])

def chargeMovie(listOfFrames, lambda3, lambda4, lambda5):
    framePairsCost = sum([chargeFramePair(listOfFrames[i], listOfFrames[i+1], lambda3) for i in \
        range(len(listOfFrames)-1)])

    frameCost = sum([chargeFrame(frame, lambda4, lambda5) for frame in listOfFrames])

    return framePairsCost + frameCost

def chargeRecoveredEltAgainstGroundTruth(convolvedElt, groundTruthElt, lambda6):
    return lambda6*(convolvedElt-groundTruthElt)**2

def chargeRecoveredFrameAgainstGroundTruth(convolvedFrame, groundTruthFrame, lambda6):
    return sum([chargeRecoveredEltAgainstGroundTruth(convolvedElt, groundTruthElt, lambda6) for \
        convolvedElt, groundTruthElt in zip(convolvedFrame, groundTruthFrame)])

def chargeGroundTruth(occluderSeq, listOfFrames, groundTruth, getConvolvedFrame, lambda6):
    convolvedFrames = [getConvolvedFrame(occluderSeq, frame) for frame in listOfFrames]

    cost = sum([chargeRecoveredFrameAgainstGroundTruth(convolvedFrame, groundTruthFrame, lambda6) for \
        convolvedFrame, groundTruthFrame in zip(convolvedFrames, groundTruth)])

    return cost

def evaluateSequenceMaker(vecBoundary, groundTruth, getConvolvedFrame, lambdas):
    def evaluateSequence(vec):
        lambda1, lambda2, lambda3, lambda4, lambda5, lambda6 = lambdas

        occluderSeq = vec[:vecBoundary]
        occluderCost = chargeOccluder(occluderSeq, lambda1, lambda2)

        movieSeq = vec[vecBoundary:]
        listOfFrames = splitMovieIntoListOfFrames(movieSeq, frameLength)
        movieCost = chargeMovie(listOfFrames, lambda3, lambda4, lambda5)

        comparisonToGroundTruthCost = chargeGroundTruth(occluderSeq, listOfFrames, groundTruth, getConvolvedFrame, \
            lambda6)

        return occluderCost + movieCost + comparisonToGroundTruthCost

    return evaluateSequence

def occGradContributionFromSingleElt(convolvedElt, groundTruthElt, lambda6, occIndex, 
    getCoefficient, hypothesizedFrame):

    return lambda6*2*(convolvedElt-groundTruthElt)*getCoefficient(occIndex, hypothesizedFrame)

def getGradientContributionFromSingleFrame(convolvedFrame, groundTruthFrame, lambda6, 
    occIndex, getCoefficient, hypothesizedFrame):

    return sum([occGradContributionFromSingleElt(convolvedElt, groundTruthElt, lambda6) for \
        convolvedElt, groundTruthElt in zip(convolvedFrame, groundTruthFrame)])

def getGradientFromObservationMatching(convolvedFrames, groundTruth, lambda6, 
    getCoefficient, hypothesizedMovie):

#    convolvedFrames = [getConvolvedFrame(occluderSeq, hypothesizedFrame) for hypothesizedFrame in listOfFrames]

    numFrames = len(hypothesizedMovie)

    gradientContribution = [sum([getGradientContributionFromSingleFrame(convolvedFrame, groundTruthFrame, 
        lambda6, occIndex, getCoefficient, hypothesizedMovie[frameIndex]) for \
        convolvedFrame, groundTruthFrame, frameIndex in zip(convolvedFrames, groundTruth, range(numFrames))]) for occIndex in range(occLength)]


    return np.array(gradientContribution)


def getOccluderGradient(convolvedFrames, groundTruth, getCoefficient, hypothesizedMovie, 
    lambda1, lambda2, lambda6):

    gradientFromObservationMatching = getGradientFromObservationMatching(convolvedFrames, groundTruth, lambda6, 
        getCoefficient, hypothesizedMovie)

    gradientFromSparsity = occGradSparsity(occVal, lambda1)
    gradientFromSpatial = occGradSpatial(occVal, lambda2)

    return gradientFromObservationMatching + gradientFromSparsity + gradientFromSpatial

def listSum(listOfLists):
    returnList = []

    for l in listOfLists:
        returnList += l

    return returnList

def frameGradContributionFromSingleElt(convolvedElt, groundTruthElt, lambda6, frameIndex, 
    getCoefficient, hypothesizedOccluder):

    return lambda6*2*(convolvedElt-groundTruthElt)*getCoefficient(frameIndex, hypothesizedOccluder)

def singleFrameGradientObservationMatching():
    return sum(frameGradContributionFromSingleElt(convolvedElt, groundTruthElt, lambda6) for \
        convolvedElt, groundTruthElt in zip(convolvedFrame, groundTruthFrame))



def frameGradientFromObservationMatching(convolvedFrames, groundTruth, getFrameCoefficient, hypothesizedMovie, lambda6):
    
    gradientContribution = []

    for convolvedFrame, groundTruthFrame, hypothesizedFrame in zip(convolvedFrames, groundTruth, hypothesizedMovie):
        gradientContribution += singleFrameGradientObservationMatching()

    return np.array(gradientContribution)

def frameGradSparsitySingleElt():
    if elt == 0:
        return 0

    elif elt > 0:
        return lamda

    else:
        return -lamda

def singleFrameGradientFromSparsity():
    return [frameGradSparsitySingleElt(elt, lamda) for elt in hypothesizedFrame]

def frameGradientFromSparsity():

    gradientContribution = []

    for hypothesizedFrame in hypothesizedMovie:
        gradientContribution += singleFrameGradientFromSparsity()

    return np.array(gradientContribution)

def occGradSpatialDoubleElt(occVal, otherOccVal, lambda2):
    if occVal == otherOccVal:
        return 0

    elif occVal > otherOccVal:
        return lambda2

    elif occVal < otherOccVal:
        return -lambda2

def occGradSpatialTripleElt(occValLeft, occValCenter, occValRight, lambda2):
    if occValLeft is None:
        return occGradSpatialDoubleElt(occValCenter, occValRight, lambda2)

    if occValRight is None:
        return occGradSpatialDoubleElt(occValCenter, occValLeft, lambda2)

    else:
        return occGradSpatialDoubleElt(occValCenter, occValRight, lambda2) + \
            occGradSpatialDoubleElt(occValCenter, occValLeft, lambda2)

def occGradSpatial(occVal, lambda2):
    return np.array([occGradSpatialTripleElt(None, occVal[0], occVal[1], lambda2)] + \
        [occGradSpatialTripleElt(occVal[i-1], occVal[i], occVal[i+1], lambda2) for i in range(1, len(occVal)-1)] + \
        [occGradSpatialTripleElt(occVal[-2], occVal[-1], None, lambda2)])


def singleFrameGradientFromSpatial(frameVal, lamb):
    pass

def frameGradientFromSpatial():

    gradientContribution = []

    for hypothesizedFrame in hypothesizedMovie:
        gradientContribution += singleFrameGradientFromSpatial()

    return np.array(gradientContribution)

def getFramesGradient(convolvedFrames, groundTruth, getFrameCoefficient, hypothesizedMovie, lambda3, lambda4, lambda5, lambda6):

    gradientFromObservationMatching = frameGradientFromObservationMatching(convolvedFrames, groundTruth, getFrameCoefficient,
        hypothesizedMovie, lambda6)

    gradientFromSparsity = frameGradientFromSparsity()

    gradientFromSpatial = frameGradientFromSpatial()


def fPrimeMaker(vecBoundary, groundTruth, lambdas):
    lambda1, lambda2, lambda3, lambda4, lambda5, lambda6 = lambdas

    occluderGradient = 1

def averageOverPolysInBinList(listOfPolynomials, binaryList):
    averagePolynomial = Polynomial([0])
    
    for i in range(len(listOfPolynomials)):
        if binaryList[i]:
            averagePolynomial += listOfPolynomials[i]
    
    return averagePolynomial / sum(binaryList)
    

def nearbyRootExists(listOfExistingRoots, root, threshold=3e-2):
    listOfExistingRootsCopy = listOfExistingRoots[:]
#    random.shuffle(listOfExistingRootsCopy)
    
    for i, existingRoot in enumerate(listOfExistingRootsCopy):
          
        if np.abs(root-existingRoot[0])<threshold:            
            return (True, i)
                        
    return (False, None)

def augmentListOfExistingRoots(listOfExistingRoots, polynomial):
    for root in polynomial.roots():
        rootNearby = nearbyRootExists(listOfExistingRoots, root)
        if rootNearby[0]:
            listOfExistingRoots[rootNearby[1]][1] += 1           
            
        else:
#            listOfExistingRoots.append([root, 1])  
            pass

def logexp(x, mu, sigma):
    return exp(-(log(x-mu))**2/(2*sigma**2))/(x*sigma*sqrt(2*pi))

def normal(x, mu, sigma):
    return exp(-(x-mu)**2/(2*sigma**2))/(sigma*sqrt(2*pi))

def laplace(x, mu, sigma):
    return exp(-abs(x-mu)/(sigma))/(2*sigma)

def logLaplace(x, mu, sigma):
    return exp(-abs(log(x)-mu)/(sigma))/(2*x*sigma)
    
def reverseLogLaplace(x, mu, sigma):
    return 2*x*sigma*exp(abs(log(x-mu))/sigma)    

def makeEvenPointsAccordingToDensity(densityFunc, numPointsParam, lowBound, upBound, startLoc, \
    maxPoints):        
    
    maxPointsUp = int((maxPoints-1)/2)    
    maxPointsDown = int((maxPoints-1)/2)
                
    upList = makeEvenPointsAccordingToDensityUnidirectional(densityFunc, \
        numPointsParam, upBound, startLoc, maxPointsUp, "up")
  
    downList = makeEvenPointsAccordingToDensityUnidirectional(densityFunc, \
        numPointsParam, lowBound, startLoc, maxPointsDown, "down")    
    
            
    return downList[::-1] + [startLoc] + upList
            
def makeEvenPointsAccordingToDensityUnidirectional(densityFunc, numPointsParam, bound, \
    startLoc, maxPoints, direction):
    
    returnList = []
    
    currentX = startLoc
    
    boundExceeded = False
    
    if direction == "up":        
        currentX += 1 / (numPointsParam * densityFunc(currentX))
        
        if currentX > bound:
            boundExceeded = True
            
    elif direction == "down":
        currentX -= 1 / (numPointsParam * densityFunc(currentX))
        
        if currentX < bound:
            boundExceeded = True      
            
    if len(returnList) >= maxPoints:
        boundExceeded = True          
    
    while not boundExceeded:        
        returnList.append(currentX)
        
        if direction == "up":
            currentX += 1 / (numPointsParam * densityFunc(currentX))
        
            if currentX > bound:
                boundExceeded = True
            
        elif direction == "down":
            currentX -= 1 / (numPointsParam * densityFunc(currentX))
        
            if currentX < bound:
                boundExceeded = True      
            
        if len(returnList) >= maxPoints:
            boundExceeded = True                  
        
    return returnList
        
    
def ryy(seq):    
    n = len(seq)

    returnSeq = [0]*n
    
    for k in range(len(seq)):
        for i in range(n - k):
            returnSeq[k] += seq[i] 
            
        returnSeq[k] /= n-k
        
    return returnSeq
    
def ryyNew(seq):
    n = len(seq)
    
    returnSeq = [0]*(2*n-1)
    
    for j in range(n):
        for k in range(-n+1, n):
            if j-k >= 0 and j-k < n:
                returnSeq[n-1+k] += seq[j]*seq[j-k]
            
    returnSeq = [i/n for i in returnSeq]        
            
    return np.array(returnSeq)
            
        
def generateImpulseSeq(n):
    return np.array([1] + [0]*(n-1))

def convolveMaker(occluder):
    return lambda frame: convolve(occluder, frame, mode="full")

def generateZeroOneSeq(n):
    oddsOfTranstion = FLIP_DENSITY/n

    if random.random() < 0.5:
        currentState = "zero"
    else:
        currentState = "one"

    returnList = []

    for _ in range(n):
        if currentState == "zero":
            if random.random() < oddsOfTranstion:
                currentState = "one"

        elif currentState == "one":
            if random.random() < oddsOfTranstion:
                currentState = "zero"

        else:
            raise

        if currentState == "zero":
            returnList.append(0)

        elif currentState == "one":
            returnList.append(1)

        else:
            raise

    return np.array(returnList)

def generateZeroOneSeqIndep(n):
    return np.array([1*(random.random()<0.5) for _ in range(n)])

def conv(sparseSeq, zeroOneSeq):
    return ifft(np.multiply(fft(sparseSeq), fft(zeroOneSeq)))

def addNoise(arr):    
    return arr + np.random.normal(loc=0, scale=NOISE_SIGMA, size=arr.shape)

#    return convolve(sparseSeq, zeroOneSeq, mode="valid")

def deconvFromZeroOne(seq, zeroOneSeq):
    result, remainder = deconvolve(seq, zeroOneSeq)
    return result

def getConjugateSeq(seq, convSeq):
    return ifft(numpyDivideNoDivZero(fft(convSeq), fft(seq)))

def displayRoots(polynomial, listOfExistingRoots, blueThresh, numPolynomialsCounted):
    listOfRepeatedRoots = []
    
    for root in polynomial.roots():
        rootNearby = nearbyRootExists(listOfExistingRoots, root)
   
#        if False:
        if rootNearby[0]:
            cleanRoot = listOfExistingRoots[rootNearby[1]]
                        
            if cleanRoot[1]/numPolynomialsCounted > blueThresh:
                x = np.real(cleanRoot[0])
                y = np.imag(cleanRoot[0])
                p.plot(x, y, "bo")
                listOfRepeatedRoots.append(cleanRoot[0])
                
            else:
                x = np.real(cleanRoot[0])
                y = np.imag(cleanRoot[0])
                p.plot(x, y, "go")                
    
        else:
            x = np.real(root)
            y = np.imag(root)
            p.plot(x, y, "ro")            
        
    unitCircle = p.Circle((0, 0), 1, color='k', fill=False)    
        
    ax = p.gca()
    
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])    
    ax.set_aspect("equal")
    ax.add_artist(unitCircle) 
        
    p.show()
    
    return listOfRepeatedRoots    

def aggregateFrames(listOfFrames, P):
    dftFrames = [np.fft.fft2(frame) for frame in listOfFrames]

    for dftFrame in dftFrames:
        viewFrame(imageify(np.abs(dftFrame)), adaptiveScaling=True,
            magnification=sqrt(dftFrame.shape[0]*dftFrame.shape[1]))

    absDftFramesToTheP = [np.power(np.abs(dftFrame), P) for \
        dftFrame in dftFrames]

    sumAbsDftFramesToTheP = np.sum(np.array(absDftFramesToTheP), axis=0)

 #   print sumAbsDftFramesToTheP.shape
    viewFrame(imageify(sumAbsDftFramesToTheP), adaptiveScaling=True,
        magnification=sqrt(sumAbsDftFramesToTheP.shape[0]*dftFrame.shape[1]))

    weightedDft = np.sum(np.array([np.divide(np.multiply(absDftFrameToTheP, \
            dftFrame), sumAbsDftFramesToTheP) for \
            absDftFrameToTheP, dftFrame in \
            zip(absDftFramesToTheP, dftFrames)]), \
            axis = 0)

#   print weightedDft[0]
    viewFrame(imageify(np.abs(weightedDft)), adaptiveScaling=True)

    return np.fft.ifft2(weightedDft)

def deconv(seq):
    
    oddsOfZeroNonzeroTransition = ZERO_DENSITY/n
    oddsOfNonzeroZeroTransition = NONZERO_DENSITY/n    
    oddsOfFlipTransition = FLIP_DENSITY/n
    
    apparentSparseSeq = seq[:]
    
    while True:
        
        sparseLikelihood = buildLikelihoodArray(apparentSparseSeq, \
            oddsOfZeroNonzeroTransition, oddsOfNonzeroZeroTransition, \
            likelihoodOfZeroGivenValSparse, likelihoodOfNonzeroGivenValSparse)        

        currentSparseSeq = snapToSparse(apparentSparseSeq, sparseLikelihood)

        viewFlatFrame(imageify(currentSparseSeq), magnification=1)

        apparentZeroOneSeq = getConjugateSeq(currentSparseSeq, seq)
                
        viewFlatFrame(imageify(apparentZeroOneSeq), magnification=1)
        
        zeroOneLikelihood = buildLikelihoodArray(apparentZeroOneSeq, \
            oddsOfFlipTransition, oddsOfFlipTransition, \
            likelihoodOfZeroGivenValZeroOne, likelihoodOfOneGivenValZeroOne)
                        
        currentZeroOneSeq = snapToZeroOne(apparentZeroOneSeq, zeroOneLikelihood)

        viewFlatFrame(imageify(currentZeroOneSeq), magnification=1)        
        
        apparentSparseSeq = getConjugateSeq(currentZeroOneSeq, seq)        
            
        viewFlatFrame(imageify(apparentSparseSeq), magnification=1)
        
def snapToZeroOne(seq, zeroLikelihood):
    returnArray = []
    
    for i, val in enumerate(seq):
        if zeroLikelihood[i] > 0.5:
            returnArray.append(0)
        else:
            returnArray.append(1)
            
    return np.array(returnArray)

def snapToSparse(seq, zeroLikelihood):
    returnArray = []

    for i, val in enumerate(seq):
        if zeroLikelihood[i] > 0.5:
            returnArray.append(0)
        else:
            returnArray.append(val)
            
    return np.array(returnArray)
    
def likelihoodOfZeroGivenValSparse(val):
    return 1/sqrt(2*pi*NOISE_SIGMA)*exp(-val**2/(2*NOISE_SIGMA**2))
    
def likelihoodOfNonzeroGivenValSparse(val):
    return 1/sqrt(2*pi*SIGNAL_SIGMA)*exp(-val**2/(2*SIGNAL_SIGMA**2))

def likelihoodOfZeroGivenValZeroOne(val):
    return 1/sqrt(2*pi)*exp(-val**2/2)
    
def likelihoodOfOneGivenValZeroOne(val):
    return 1/sqrt(2*pi)*exp(-(val-1)**2/2)    

def buildLikelihoodArray(seq, oddsOfZeroNonzeroTransition, oddsOfNonzeroZeroTransition, \
    likelihoodOfZeroGivenVal, likelihoodOfNonzeroGivenVal):
    
    forwardArray = buildOneDirectionalLikelihoodArray(seq, oddsOfZeroNonzeroTransition, 
        oddsOfNonzeroZeroTransition, likelihoodOfZeroGivenVal, likelihoodOfNonzeroGivenVal, "forward")
        
    backwardArray = buildOneDirectionalLikelihoodArray(seq, oddsOfZeroNonzeroTransition, 
        oddsOfNonzeroZeroTransition, likelihoodOfZeroGivenVal, likelihoodOfNonzeroGivenVal, "backward")
        
    return np.array([(i*j)/(i*j+(1-i)*(1-j)) for i,j in zip(forwardArray, backwardArray)])

def buildOneDirectionalLikelihoodArray(seq, oddsOfZeroNonzeroTransition, oddsOfNonzeroZeroTransition, \
    likelihoodOfZeroGivenVal, likelihoodOfNonzeroGivenVal, direction):
        
    startPZero = ZERO_DENSITY / (ZERO_DENSITY + NONZERO_DENSITY)

    n = len(seq)

    likelihoodWhereLastEntryIsZero = [startPZero]
    likelihoodWhereLastEntryIsNonzero = [1-startPZero]

    if direction == "forward":
        i = 0
    elif direction == "backward":
        i = n-1
    else:
        raise

    while 0 <= i and i < n:
        val = seq[i]
        
        if direction == "forward":
            lastEntryZero = likelihoodWhereLastEntryIsZero[-1]
            lastEntryNonzero = likelihoodWhereLastEntryIsNonzero[-1]
        elif direction == "backward":
            lastEntryZero = likelihoodWhereLastEntryIsZero[0]
            lastEntryNonzero = likelihoodWhereLastEntryIsNonzero[0]
        else: 
            raise

        zeroToZero = lastEntryZero * (1-oddsOfZeroNonzeroTransition) * \
            likelihoodOfZeroGivenVal(val)

        zeroToNonzero = lastEntryZero * (oddsOfZeroNonzeroTransition) * \
            likelihoodOfNonzeroGivenVal(val)

        nonzeroToZero = lastEntryNonzero * (oddsOfNonzeroZeroTransition) * \
            likelihoodOfZeroGivenVal(val)

        nonzeroToNonzero = lastEntryNonzero * (1-oddsOfNonzeroZeroTransition) * \
            likelihoodOfNonzeroGivenVal(val)

        nextEntryZero = zeroToZero + nonzeroToZero
        nextEntryNonzero = zeroToNonzero + nonzeroToNonzero

        nextEntryZeroNormalized = nextEntryZero / (nextEntryZero + \
            nextEntryNonzero)
            
        nextEntryNonzeroNormalized = nextEntryNonzero / (nextEntryZero + \
            nextEntryNonzero)        

        if direction == "forward":
            likelihoodWhereLastEntryIsZero.append(nextEntryZeroNormalized)
            likelihoodWhereLastEntryIsNonzero.append(nextEntryNonzeroNormalized)
            i += 1
        elif direction == "backward":
            likelihoodWhereLastEntryIsZero = [nextEntryZeroNormalized] + likelihoodWhereLastEntryIsZero
            likelihoodWhereLastEntryIsNonzero = [nextEntryNonzeroNormalized] + likelihoodWhereLastEntryIsNonzero
            i -= 1
        else:
            raise            



    return likelihoodWhereLastEntryIsZero
        

def getListOfPolynomialsFromConvolvedDifferenceFrames(listOfConvolvedDifferenceFrames):
    listOfPolynomials = []
    
    for differenceFrame in listOfConvolvedDifferenceFrames:
        swappedFrame = np.swapaxes(differenceFrame, 0, 1)
        
        for singleColorFrame in swappedFrame:
            listOfPolynomials.append(Polynomial(singleColorFrame))
            
    return listOfPolynomials

def getListOfSingleColorFrames(listOfFrames):
    listOfSingleColorFrames = []
    
    for frame in listOfFrames:
        swappedFrame = np.swapaxes(frame, 0, 1)
        
        for singleColorFrame in swappedFrame:
            listOfSingleColorFrames.append(singleColorFrame)
            
    return listOfSingleColorFrames
    
def getSingleColorFrames(frame):
    listOfSingleColorFrames = []
    
    swappedFrame = np.swapaxes(frame, 0, 1)
    
    for singleColorFrame in swappedFrame:
        listOfSingleColorFrames.append(singleColorFrame)
            
    return listOfSingleColorFrames    

def visualizePolynomialValues(seq, numSteps=300, minX=-1.5, maxX=1.5,
     minY=-1.5, maxY=1.5):

    n = len(seq)

    xRange = np.linspace(minX, maxX, numSteps)
    yRange = np.linspace(minY, maxY, numSteps)
        
    X, Y = np.meshgrid(xRange, yRange)
        
#    Z = np.log(np.abs(np.polyval(seq, X + 1j*Y)))
    
#    p.pcolormesh(X, Y, Z, cmap=cm.gist_rainbow)
#    p.colorbar()
#    p.show()
    
#    Z = np.log(np.abs(np.polyval(seq, \
#        np.exp(2*pi*1j*random.random())*np.abs(X + 1j*Y))))
    
#    p.pcolormesh(X, Y, Z, cmap=cm.gist_rainbow)
#    p.colorbar()
#    p.show()
        
        
    absMesh = np.abs(X + 1j*Y)

#   This corresponds to the magnitude of the all-ones polynomial 
    logNormalizationMesh = np.log(np.polyval(np.array([1]*n), absMesh))

    
#    Z = np.log(np.divide(np.abs(np.polyval(seq, X + 1j*Y)), \
#        np.abs(np.polyval(seq, np.exp(2*pi*1j*random.random())*np.abs(X + 1j*Y)))))
 
    Z = np.log(np.abs(np.polyval(seq, X + 1j*Y))) - \
        logNormalizationMesh
        
        
#    Z = X + 1j*Y
            
    visualizeColorMesh(X, Y, Z, np.roots(seq))
    visualizeColorMesh(X, Y, Z, [])

def findRandomAcceptableDirection(sortedListOfDirectionsWithRoots, minimumDistance, maxFailures=50):
    randomAngle = 2*pi*random.random()

    if maxFailures == 0:
        print "Warning: could not find an acceptable direction. Lower min distance to fix issue."
        return randomAngle
    
    index = np.searchsorted(sortedListOfDirectionsWithRoots, randomAngle)
    
    n = len(sortedListOfDirectionsWithRoots)

    leftAngle = sortedListOfDirectionsWithRoots[(index-1)%n]
    rightAngle = sortedListOfDirectionsWithRoots[index%n]
    
    if ((abs(randomAngle-leftAngle) > minimumDistance) and \
        (abs(randomAngle-rightAngle) > minimumDistance)):

#        print 50 - maxFailures

        return randomAngle
         
    else:
        return findRandomAcceptableDirection(sortedListOfDirectionsWithRoots, minimumDistance, \
            maxFailures - 1)    

def makePolynomialMesh(seq, normalizationAngle, numSteps=300, minX=-1.5, maxX=1.5,
     minY=-1.5, maxY=1.5):  
    
    xRange = np.linspace(minX, maxX, numSteps)
    yRange = np.linspace(minY, maxY, numSteps)
        
    X, Y = np.meshgrid(xRange, yRange)    

    absMesh = np.abs(X + 1j*Y)

#   This corresponds to the magnitude of the all-ones polynomial 
#    normalizationMesh = np.polyval(np.abs(seq), absMesh)


#    print len(seq), len(sortedListOfDirectionsWithRoots)

#    sortedListOfDirectionsWithRoots = sorted([cis(root) for root in np.roots(seq)])

#    minimumDistance = 2*pi/(len(seq)*7)

#    randomAngle = findRandomAcceptableDirection(sortedListOfDirectionsWithRoots, \
#        minimumDistance)

    normalizationMesh = np.abs(np.polyval(seq, np.multiply(absMesh, np.exp(1j*normalizationAngle))))



#    visualizeColorMesh(X, Y, np.log(normalizationMesh), [])
#    visualizeColorMesh(X, Y, np.log(np.abs(np.polyval(seq, X + 1j*Y))), [])

    heuristicReward = np.vectorize(heuristicRewardFunc)


    Z = np.log(np.divide(np.abs(np.polyval(seq, X + 1j*Y)), \
        normalizationMesh)+1) \
        + heuristicReward(X+1j*Y)
 
#    Z = np.log(np.abs(np.polyval(seq, X + 1j*Y))+1)
         
  
    return X, Y, Z
    
#    yRange = 
          
def makeRootMagnitudeHistogram(listOfSeqs, sigma):
#    hist1, hist2 = np.histogram([np.abs(r) for r in np.roots(seq)], bins=30)
    
    listOfRoots = []
    
    for seq in listOfSeqs:
        listOfRoots.extend(np.roots(seq))
    
    p.hist([np.abs(r) for r in listOfRoots], bins=200, range=(0.5, 2), density=True)
    
#    sigmaRange = np.linspace(0.03, 0.03, 1)
    xRange = np.linspace(0.01, 3, 100)
    
#    for sigma in sigmaRange:
 #       p.plot(xRange, [logexp(i, 0, sigma) for i in xRange])
 #       p.plot(xRange, [normal(i, 1, sigma) for i in xRange])
#        p.plot(xRange, [laplace(i, 1, sigma) for i in xRange])    
 
    p.plot(xRange, [logLaplace(i, 0, sigma) for i in xRange])  
  
    p.axvline(x=1, color="k")
    p.show()
    
        
def heuristicRewardFunc(x):
#    lambda1 = -0.3
    lambda1 = 0

    awayFromUnitCircle = lambda1 * sqrt(abs(abs(x) - 1))

#    lambda2 = -0.3 
    lambda2 = 0

    awayFromRealLine = lambda2 * abs(np.imag(x))**(1/4)

    return awayFromRealLine + awayFromUnitCircle

def makeFuzzyPolynomialMesh(seq, numSteps=300, minX=-1.5, maxX=1.5,
    minY=-1.5, maxY=1.5, numSamples=100):
     
    averageZ = 0
     
    for _ in range(numSamples):
        noisySeq = addNoise(seq)
         
#        displayRoots(Polynomial(noisySeq[::-1]),\
#            [], 0, 1) 
         
        X, Y, Z = makePolynomialMesh(noisySeq, numSteps=300, minX=-1.5, maxX=1.5,\
            minY=-1.5, maxY=1.5)
        
        averageZ += Z
#        averageZ = np.logaddexp(averageZ, Z)
             
    averageZ /= numSamples
    
    return X, Y, averageZ     
    
def makeAggregateMesh(listOfSeqs, normalizationAngle, numSteps=300, minX=-1.5, maxX=1.5,
    minY=-1.5, maxY=1.5):
    
    averageZ = 0
    
    for i, seq in enumerate(listOfSeqs):
        
        if i % 100 == 0:
            pront(str(i) + "/" + str(len(listOfSeqs)))
            
        
        X, Y, Z = makePolynomialMesh(seq, normalizationAngle, numSteps=300, minX=-1.5, maxX=1.5,\
            minY=-1.5, maxY=1.5)    
            
        averageZ += Z
        
    averageZ /= len(listOfSeqs)
    
    return X, Y, averageZ
#    for _ in range()    

def makeRadialMesh(seq, normalizationAngle, densityFunc, thetaSteps=1000, rSteps=1000):
    
    thetaRange = np.linspace(0, 2*pi, thetaSteps)    
    
    rPointsParam = rSteps/10
    
    rRange = makeEvenPointsAccordingToDensity(densityFunc, rPointsParam, 0, 10, 1, \
        rSteps*2)

    THETA, R = np.meshgrid(thetaRange, rRange)
    
#    print R, THETA
    
    absMesh = R

#    print len(seq), len(sortedListOfDirectionsWithRoots)

#    sortedListOfDirectionsWithRoots = sorted([cis(root) for root in np.roots(seq)])

#    minimumDistance = 2*pi/(len(seq)*7)

#    randomAngle = findRandomAcceptableDirection(sortedListOfDirectionsWithRoots, \
#        minimumDistance)

    normalizationMesh = np.abs(np.polyval(seq, np.multiply(absMesh, np.exp(1j*normalizationAngle))))

#    Z = np.log(np.divide(np.abs(np.polyval(seq, np.multiply(np.exp(1j*THETA), R))), \
#        np.abs(np.polyval(seq, np.exp(2*pi*1j*random.random())*R)))+1)

    Z = np.log(np.divide(np.abs(np.polyval(seq, np.multiply(np.exp(1j*THETA), R))), \
        normalizationMesh)+1)    

    rProxyRange = np.linspace(0, 1, len(rRange))    
    THETA_PROXY, R_PROXY = np.meshgrid(thetaRange, rRange)

    return THETA_PROXY, R_PROXY, Z

    
def makeAggregateRadialMesh(listOfSeqs, normalizationAngle, densityFunc, thetaSteps=1000, rSteps=1000):
    
    averageZ = 0
    
    for i, seq in enumerate(listOfSeqs):
        
        if i % 100 == 0:
            pront(str(i) + "/" + str(len(listOfSeqs)))
            
        
        THETA, R, Z = makeRadialMesh(seq, normalizationAngle, densityFunc, thetaSteps, rSteps)        
            
        averageZ += Z
        
    averageZ /= len(listOfSeqs)
    
    return THETA, R, averageZ

def cumLogLaplaceMaker(mu, sigma):
    def cumLogLaplaceParametrized(y):
        if y <= 0:
            return 0
        else:
            return (1 + np.sign(np.log(y)-mu)*(1-np.exp(-np.abs(np.log(y)-mu)/sigma)))/2
    return cumLogLaplaceParametrized

def warpROld(protoR, mu, sigma):
    print sigma

#    sigma=0.2

    cumLogLaplaceParametrized = cumLogLaplaceMaker(mu, sigma)

    inverseCumLogLaplace = inversefunc(cumLogLaplaceParametrized, domain=[1e-5, float("Inf")])

    r = inverseCumLogLaplace(protoR)

    print "protoR", protoR
    print "r", r
    print "should be equal to protoR", cumLogLaplaceParametrized(r)

    return r

def warpR(protoR, evenPoints): 

#    print len(evenPoints), protoR*(len(evenPoints)-1)

    return fuzzyLookup(evenPoints, protoR*(len(evenPoints)-1))


def evaluatePointForSingleSeqRadial(seq, normalizationAngle, theta, r):
#    sortedListOfDirectionsWithRoots = sorted([cis(root) for root in np.roots(seq)])

#    minimumDistance = 2*pi/(len(seq)*7)

#    randomAngle = findRandomAcceptableDirection(sortedListOfDirectionsWithRoots, \
#        minimumDistance)

    normalization = np.abs(np.polyval(seq, r*np.exp(1j*normalizationAngle)))

#    Z = np.log(np.divide(np.abs(np.polyval(seq, np.multiply(np.exp(1j*THETA), R))), \
#        np.abs(np.polyval(seq, np.exp(2*pi*1j*random.random())*R)))+1)

    z = np.log(np.abs(np.polyval(seq, r*np.exp(1j*theta))) / normalization+1)        

    return z

def evaluatePointForSingleSeq(seq, normalizationAngle, x, y):

    r = np.abs(x+1j*y)

    normalization = np.abs(np.polyval(seq, r*np.exp(1j*normalizationAngle)))

    z = np.log(np.abs(np.polyval(seq, x+1j*y)) / normalization+1)        

    return z    

def evaluatePointMakerRadial(listOfSeqs, normalizationAngle, evenPoints, mu, sigma, verbose=False):
    def evaluatePoint(thetaRArray):  
        theta = thetaRArray[0]
        protoR = thetaRArray[1]

        if protoR <= 0:
            returnVal = 1e4*(protoR-1)**2

            if verbose:
                print theta, protoR, returnVal

            return returnVal

        if protoR >= 1:
            returnVal = 1e4*(protoR+2)**2

            if verbose:
                print theta, protoR, returnVal

            return returnVal

        r = warpR(protoR, evenPoints)

#        print "theta, r, protoR", theta, r, protoR

        output.write("theta " + str(theta))
        output.write("r " + str(r))

        averageZ = 0 

        for i, seq in enumerate(listOfSeqs):
            z = evaluatePointForSingleSeq(seq, normalizationAngle, theta, r)

            averageZ += z

        averageZ /= len(listOfSeqs)

        if verbose:
            print theta, protoR, averageZ

        return averageZ

    return evaluatePoint

def evaluatePointMaker(listOfSeqs, normalizationAngle, evenPoints, mu, sigma, verbose=True):
    def evaluatePoint(XYArray):  
        x = XYArray[0]
        y = XYArray[1]

#        p.plot(x, y, "w.")

        averageZ = 0 

        for i, seq in enumerate(listOfSeqs):
            z = evaluatePointForSingleSeq(seq, normalizationAngle, x, y)

            averageZ += z

        averageZ /= len(listOfSeqs)

        if verbose:
            print x, y, averageZ

        return averageZ

    return evaluatePoint

def getGradient(f, currentVal, point, epsilon):
    d = len(point)

    gradient = []

    for i in range(d):
        slightlyDifferentPoint = np.array(point[:])

        slightlyDifferentPoint[i] += epsilon

        gradient.append(-(f(slightlyDifferentPoint) - currentVal)/epsilon)

    gradient = np.array(gradient)

    magnitude = np.linalg.norm(gradient)

    return gradient/magnitude, magnitude

def rewardFunctionMaker(f, currentPoint, gradientDirection, epsilon):
    def rewardFunction(distance):
        
        distanceVal = f(currentPoint + distance*gradientDirection)
        distanceDerivative = (f(currentPoint + (distance+epsilon)*gradientDirection) - \
            distanceVal)/epsilon

        return distanceVal, distanceDerivative

    return rewardFunction

def findOptimalDistanceBetweenTwoPoints(rewardFunction, gtol, point1, \
    point2, bestVal, bestDistance, maxiter = 30):

#    print "point1", "point2", point1, point2

    inBetweenPoint = (point1 + point2)/2

    currentVal, currentDerivative = rewardFunction(inBetweenPoint)

    if maxiter == 0:
        return None, None

    if currentVal < bestVal:
        bestVal = currentVal
        bestDistance = inBetweenPoint

    if abs(currentDerivative) < gtol:
        return bestVal, bestDistance

    if currentDerivative <= 0:
        return findOptimalDistanceBetweenTwoPoints(rewardFunction, gtol, inBetweenPoint, \
            point2, bestVal, bestDistance, maxiter - 1)

    else:
        return findOptimalDistanceBetweenTwoPoints(rewardFunction, gtol, point1, \
            inBetweenPoint, bestVal, bestDistance, maxiter - 1)

def findOptimalDistance(rewardFunction, gtol, initDistance=1e-2, maxiter=8):


#    zeroVal, zeroDerivative = rewardFunction(0)

#    print "zeroVal", zeroVal 
#    print "zeroDerivative", zeroDerivative

#    assert zeroDerivative < 0                


    currentDistance = initDistance
    oldCurrentDistance = 0

    currentVal, currentDerivative = rewardFunction(initDistance)

    bestVal = currentVal
    bestDistance = currentDistance

    iterCount = 0

    while currentDerivative <= 0:
#        print "currentDistance", currentDistance

        oldCurrentDistance = currentDistance
        currentDistance *= 2

        currentVal, currentDerivative = rewardFunction(currentDistance)

        if currentVal < bestVal:
            bestVal = currentVal
            bestDistance = currentDistance

        iterCount += 1
        if iterCount > maxiter:
            return None, None

    otherBestVal, otherBestDistance = findOptimalDistanceBetweenTwoPoints(rewardFunction, gtol, \
        oldCurrentDistance, currentDistance, bestVal, bestDistance)

    if otherBestVal is not None:
        if otherBestVal < bestVal:
            bestVal = otherBestVal
            bestDistance = otherBestDistance

    else:
        return None, None

    return bestDistance, bestVal

def homebrewMinimizer(f, x0, gtol=1e-3, epsilon=1e-8, maxiter=40, dtol=1e-5):
    currentPoint = x0[:]
    gradientMagnitude = gtol + 1

    currentVal = f(currentPoint)

    iterCount = 0

    while True:

        gradientDirection, gradientMagnitude = getGradient(f, \
            currentVal, currentPoint, epsilon)

        if gradientMagnitude < gtol:
            break

#        print "grad", gradientDirection, gradientMagnitude

        rewardFunction = rewardFunctionMaker(f, currentPoint, gradientDirection, epsilon)

        optimalDistance, currentVal = findOptimalDistance(rewardFunction, gtol)

        if optimalDistance < dtol:
            break

        if optimalDistance is None:
            return None, None

        currentPoint += gradientDirection*optimalDistance

        print "currentPoint", currentPoint, currentVal

        iterCount += 1

        if iterCount > maxiter:
            return None, None

    return currentPoint, currentVal

#    func_calls, f = wrap_function(f, args)
#    grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))



def findNearbyRoot(listOfSeqs, normalizationAngle, evenPoints, startingPoint, mu, sigma):
    evaluatePoint = evaluatePointMaker(listOfSeqs, normalizationAngle, evenPoints, mu, sigma, verbose=False)

    def callback(x):
        output.write("hi" + str(x))

    gtol = 1e-5

#    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = \
#        fmin_bfgs(evaluatePoint, startingPoint, epsilon=1e-5, full_output=True, gtol=1e-5, \
#        amax=1e-2)

#    print "xopt", xopt
#    print "fopt", fopt
#    print "gopt", gopt
#    print "Bopt", Bopt
#    print "func_calls", func_calls
#    print "grad_calls", grad_calls
#    print "warnflag", warnflag

#    xopt, fopt, func_calls, grad_calls, warnflag = \
#        fmin_cg(evaluatePoint, startingPoint, epsilon=1e-10, full_output=True, gtol=1e-5)

#    print "xopt", xopt
#    print "fopt", fopt
#    print "func_calls", func_calls
#    print "grad_calls", grad_calls
#    print "warnflag", warnflag

#    if np.linalg.norm(gopt) < 1e-1:
#        return xopt

#    xopt, nfeval, rc = \
#        fmin_tnc(evaluatePoint, startingPoint, epsilon=1e-8, approx_grad=1, \
#            stepmx=1e-10)

#    print "xopt", xopt
#    print "nfeval", nfeval
#    print "rc", rc

    xopt, fopt = homebrewMinimizer(evaluatePoint, startingPoint)

    print "xopt", xopt
    print "fopt", fopt

    if xopt is not None:
        return xopt, fopt

    else:
        return None, None

#    if warnflag == 0:
#        return xopt

#    else:
#        return None

def rootAlreadyFound(foundRoots, root, threshold):
    for foundRoot in foundRoots:
        if np.linalg.norm(root - foundRoot) < threshold:
            return True

    return False

def getNormalizationAngleRandom(listOfSeqs):
    seq = listOfSeqs[random.randint(0, len(listOfSeqs)-1)]

    sortedListOfDirectionsWithRoots = sorted([cis(root) for root in np.roots(seq)])

    minimumDistance = 2*pi/(len(seq)*2)

    randomAngle = findRandomAcceptableDirection(sortedListOfDirectionsWithRoots, \
        minimumDistance)

    return randomAngle

def findIndexOfGreatestGap(sortedListOfDirectionsWithRoots):
    listOfGaps = [(sortedListOfDirectionsWithRoots[i+1] - sortedListOfDirectionsWithRoots[i], \
        i) for i in range(len(sortedListOfDirectionsWithRoots)-1)]

    return sorted(listOfGaps, key=lambda x: x[0])[-1][1]


def getNormalizationAngleDistant(listOfSeqs):
    seq = listOfSeqs[random.randint(0, len(listOfSeqs)-1)]

    sortedListOfDirectionsWithRoots = sorted([cis(root) for root in np.roots(seq)])

    index = findIndexOfGreatestGap(sortedListOfDirectionsWithRoots)

    normalizationAngle = (sortedListOfDirectionsWithRoots[index] + \
            sortedListOfDirectionsWithRoots[index+1])/2

    # angles on the real line are always screwed up because many polynomials have real roots
    if abs(normalizationAngle) < 1e-4:
        return getNormalizationAngle(listOfSeqs)

    if abs(normalizationAngle - pi) < 1e-4:
        return getNormalizationAngle(listOfSeqs)

    if abs(normalizationAngle - 2*pi) < 1e-4:
        return getNormalizationAngle(listOfSeqs)

    return normalizationAngle

def warpPoint(point, evenPoints):
 #   print point

    return np.array([point[0], warpR(point[1], evenPoints)])

def findRootsRadial(listOfSeqs, correctRoots=[], showy=False):
    foundRoots = []
    gridPoints = []

    numSeqs = len(listOfSeqs)
    n = len(listOfSeqs[0])

    numThetaGridPoints = 2*n
#    numThetaGridPoints = n/8 
    numRGridPoints = int(log(n))
    threshold = 1e-3/n

    first = 2*pi/(numThetaGridPoints*2)
    second = 2*pi*(1-1/(numThetaGridPoints*2))
    third = numThetaGridPoints

    thetaGridPoints = np.linspace(first, second, third)
    rGridPoints = np.linspace(0.1, 0.9, numRGridPoints)

    normalizationAngle = getNormalizationAngleDistant(listOfSeqs[random.randint(0, numSeqs-1)])

    mu = 0
    sigma = 0.3/sqrt(n)

    rSteps = 1000
    rPointsParam = rSteps/10

    densityFunc = lambda x: logLaplace(x, 0, sigma) 

    evenPoints = makeEvenPointsAccordingToDensity(densityFunc, rPointsParam, 0, 10, 1, \
        rSteps*2)

    THETA, R, Z_RADIAL = makeAggregateRadialMesh(listOfSeqs, normalizationAngle, densityFunc)

    X, Y, Z = makeAggregateMesh(listOfSeqs, normalizationAngle, densityFunc)

#    visualizeRadialColorMesh(THETA, R, Z, sigma, [])
    visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, correctRoots)
    visualizeColorMesh(X, Y, Z, correctRoots)

    for thetaGridPoint in thetaGridPoints:
        for rGridPoint in rGridPoints:
            if radial:
                startingPoint = np.array([thetaGridPoint, rGridPoint])
            else:
                startingPoint = np.array([thetaGridPoint, warpR(rGridPoint, evenPoints)])

            print "startingPoint", startingPoint

            gridPoints.append(startingPoint)

 #           print thetaGridPoint, warpR(rGridPoint, evenPoints)

#            try:
            nearbyRoot = findNearbyRoot(listOfSeqs, normalizationAngle, evenPoints, startingPoint, mu, sigma)
#            except Exception as e:
#                print "convergence error"
#                print str(e)
#                e.printStackTrace()
 #               raise e
 #               nearbyRoot = None

            if type(nearbyRoot) != type(None):

                if nearbyRoot[0] < 0 or nearbyRoot[0] > 2*pi:
                    nearbyRoot[0] = nearbyRoot[0] % (2*pi)


                if not rootAlreadyFound(foundRoots, nearbyRoot, threshold):
                    print "found a new root!", nearbyRoot
                    if showy:
                        visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [(warpPoint(nearbyRoot, evenPoints), "b"), \
                            (warpPoint(startingPoint, evenPoints), "r")])

                    foundRoots.append(nearbyRoot)

                    if showy:
                        visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [warpPoint(i, evenPoints) for i in foundRoots])

                else:

                    if showy:
                        visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [(warpPoint(nearbyRoot, evenPoints), "b"), \
                            (warpPoint(startingPoint, evenPoints), "r")])
                    print "duplicated existing root", nearbyRoot

                    if showy:
                        visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [warpPoint(i, evenPoints) for i in foundRoots])

            else:
                print "Search failed."

    visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, correctRoots, "correct_roots.png")
    visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [(warpPoint(i, evenPoints), "m") for i in gridPoints], \
        "grid_points.png")
    visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [(warpPoint(i, evenPoints), "y") for i in foundRoots], \
        "found_roots.png")
    visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [], \
        "no_distractions.png")

    return foundRoots

def convertToXY(thetaRArray):
    theta = thetaRArray[0]
    r = thetaRArray[1]
    return np.array([r*cos(theta), r*sin(theta)])

def convertToThetaR(XYArray):
    x = XYArray[0]
    y = XYArray[1]
    return np.array([cis(x+1j*y), np.abs(x+1j*y)])

def cleanFoundRoots(sortedFoundRoots, numRootsExpected):
    prunedRoots = sortedFoundRoots[:]

    delVals = {}
    delIndices = []

#    print sortedFoundRoots

    for pr in prunedRoots:
        print pr

    for i, root1 in enumerate(sortedFoundRoots[:]):
        neighborFound = False

        if abs(root1) < 0.5:
            delIndices.append(i)

        elif abs(root1) > 2:
            delIndices.append(i)

        for j, root2 in enumerate(sortedFoundRoots[:]):
#            print i, j
            if abs(root1 - root2) < 2*pi/(numRootsExpected*4) and root1 != root2:
                if root1 > root2 and not root1 in delVals:
                    print root1, ">", root2
                    print "adding", i

                    delVals[root1] = True
                    delIndices.append(i)
                elif root2 >= root1 and not root2 in delVals and not j in delIndices:
                    print root2, ">", root1
                    print "adding", j

                    delVals[root2] = True
                    delIndices.append(j)

    print delIndices

    for delIndex in sorted(delIndices)[::-1]:
        print "deleting root", prunedRoots[delIndex]
        del prunedRoots[delIndex]

    if len(sortedFoundRoots) > numRootsExpected:
        recoveredRoots = prunedRoots[:numRootsExpected]
    else:
        recoveredRoots = prunedRoots[:]

    return recoveredRoots


def findRoots(listOfSeqs, correctRoots=[], showy=False, numRootsExpected=None):
    foundRoots = []
    annotatedFoundRoots = []

    gridPoints = []

    numSeqs = len(listOfSeqs)
    n = len(listOfSeqs[0])

    if numRootsExpected is None:
        numRootsExpected = n

    numThetaGridPoints = 2*n
  #  numThetaGridPoints = n/2
#    numThetaGridPoints = n/8 
#    numThetaGridPoints = 2

    numRGridPoints = int(log(n))    
#    numRGridPoints = 2

    threshold = 3e-2/n

    first = 2*pi/(numThetaGridPoints*2)
    second = 2*pi*(1-1/(numThetaGridPoints*2))
    third = numThetaGridPoints

    thetaGridPoints = np.linspace(first, second, third)
    rGridPoints = np.linspace(0.1, 0.9, numRGridPoints)

#    normalizationAngle = getNormalizationAngleDistant(listOfSeqs)
    normalizationAngle = getNormalizationAngleRandom(listOfSeqs)
#    normalizationAngle = pi/12
#    normalizationAngle = 5*pi/12
    

    mu = 0
    sigma = 0.3/sqrt(n)

    rSteps = 1000
    rPointsParam = rSteps/10

    densityFunc = lambda x: logLaplace(x, 0, sigma) 

    evenPoints = makeEvenPointsAccordingToDensity(densityFunc, rPointsParam, 0, 10, 1, \
        rSteps*2)

#    THETA, R, Z_RADIAL = makeAggregateRadialMesh(listOfSeqs, normalizationAngle, densityFunc)



    X, Y, Z = makeAggregateMesh(listOfSeqs, normalizationAngle, densityFunc)

#    visualizeRadialColorMesh(THETA, R, Z, sigma, [])
#    visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, correctRoots)
    print normalizationAngle
    visualizeColorMesh(X, Y, Z, [(correctRoots, "g")], normalizationAngle=normalizationAngle)

    for thetaGridPoint in thetaGridPoints:
        for rGridPoint in rGridPoints:

            startingPoint = convertToXY(np.array([thetaGridPoint, warpR(rGridPoint, evenPoints)]))

            print "startingPoint", startingPoint

            gridPoints.append(startingPoint.copy())

#            print "gridPoints", gridPoints
        

            nearbyRoot, rootVal = findNearbyRoot(listOfSeqs, normalizationAngle, \
                evenPoints, startingPoint, mu, sigma)

            if type(nearbyRoot) != type(None) and rootVal != None:

                if not rootAlreadyFound(foundRoots, nearbyRoot, threshold):
                    print "found a new root!", nearbyRoot, rootVal
                    if showy:
                        visualizeColorMesh(X, Y, Z, [(nearbyRoot, "b"), \
                            (startingPoint, "r")])

                    foundRoots.append(nearbyRoot)
                    annotatedFoundRoots.append((nearbyRoot, rootVal))

                    if showy:
                        visualizeColorMesh(X, Y, Z, [(i, "y") for i in foundRoots])

                else:

                    if showy:
                        visualizeColorMesh(X, Y, Z, [(nearbyRoot, "b"), \
                            (startingPoint, "r")])
                    print "duplicated existing root", nearbyRoot

                    if showy:
                        visualizeColorMesh(X, Y, Z, [(i, "y") for i in foundRoots])

            else:
                print "Search failed."
                if showy:
                    visualizeColorMesh(X, Y, Z, [(startingPoint, "r")])


    sortedFoundRoots = sorted(annotatedFoundRoots, key=lambda x: x[1])

    complexFormRootsFound = [root[0][0] + 1j*root[0][1] for root in sortedFoundRoots]

    pickle.dump(sortedFoundRoots, open("found_roots", "w"))

    print sortedFoundRoots

    recoveredRoots = cleanFoundRoots(complexFormRootsFound, numRootsExpected)


#    print gridPoints

    visualizeColorMesh(X, Y, Z, correctRoots, filename="correct_roots.png")
    visualizeColorMesh(X, Y, Z, [(i, "m") for i in gridPoints], filename="grid_points.png")
    visualizeColorMesh(X, Y, Z, [(i, "r") for i in recoveredRoots], filename="recovered_roots.png")
    visualizeColorMesh(X, Y, Z, [(i, "y") for i in foundRoots], filename="found_roots.png")
    visualizeColorMesh(X, Y, Z, [], filename="no_distractions.png")

 #   visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [i for i in correctRoots], "correct_roots_radial.png")
 #   visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [(convertToThetaR(i), "m") for i in gridPoints], \
 #       "grid_points_radial.png")
 #   visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [(convertToThetaR(i), "y") for i in foundRoots], \
 #       "found_roots_radial.png")
 #   visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, [], \
 #       "no_distractions_radial.png")



    return recoveredRoots

def findRootsArgRel(listOfSeqs, correctRoots=[]):
    foundRoots = []
    gridPoints = []

    numSeqs = len(listOfSeqs)
    n = len(listOfSeqs[0])

    normalizationAngle = getNormalizationAngle(listOfSeqs[random.randint(0, numSeqs-1)])

    mu = 0
    sigma = 0.3/sqrt(n)

    rSteps = 1000
    rPointsParam = rSteps/10

    densityFunc = lambda x: logLaplace(x, 0, sigma) 

    evenPoints = makeEvenPointsAccordingToDensity(densityFunc, rPointsParam, 0, 10, 1, \
        rSteps*2)

    THETA, R, Z_RADIAL = makeAggregateRadialMesh(listOfSeqs, normalizationAngle, densityFunc)

#    X, Y, Z = makeAggregateMesh(listOfSeqs, normalizationAngle, densityFunc)

#    visualizeRadialColorMesh(THETA, R, Z, sigma, [])
    visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, correctRoots)
#    visualizeColorMesh(X, Y, Z, correctRoots)

    zRadialShape = Z_RADIAL.shape

    rootsUnprocessed = argrelextrema(Z_RADIAL, np.less, order=10)

    foundRoots = [warpPoint(np.array([2*pi*theta/zRadialShape[1], r/zRadialShape[0]]), evenPoints) \
        for theta, r in zip(rootsUnprocessed[1], rootsUnprocessed[0])]

    visualizeRadialColorMesh(THETA, R, Z_RADIAL, sigma, foundRoots)

    print foundRoots

    print zRadialShape

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

def cis(val):
    angle = np.angle(val, deg=False)

    if angle >= 0:
        return angle
    else:
        return angle + 2*pi

def visualizeRadialColorMesh(THETA, R, Z, sigma, roots, filename=None):
    
    p.clf()

    print Z.shape
    
    cdict5 = createCMapDictHelix(10)

    helix = LinearSegmentedColormap("helix", cdict5)

    p.register_cmap(cmap=helix)
    
    p.pcolormesh(THETA, R, Z, cmap=helix)

    p.colorbar()
    
    for root in roots:
        if type(root) == type((0,1)):
            actualRoot = root[0]
            color = root[1]

            print "root", actualRoot, "color", color

            if type(actualRoot) == type(np.array([0,1])) and len(actualRoot) == 2:

                p.plot(actualRoot[0], np.abs(actualRoot[1]), color + ".")

            else:

                p.plot(cis(actualRoot), np.abs(actualRoot), color + ".")

        else:

            print "root", root

            if type(root) == type(np.array([0,1])) and len(root) == 2:

                p.plot(root[0], np.abs(root[1]), "c.")

            else:

#            p.plot(cis(root), np.abs(root), "w.")
                p.plot(cis(root), np.abs(root), "c.")
        
    p.axhline(y=1, color="k")
    
    ax = p.gca()
    
#    ax.set_xlim([0, 2*pi])
#    ax.set_ylim([0, 10])

    ax.set_ylim(0.001, 10)
    ax.set_yscale("log_laplace", mu=0, sigma=sigma)
    ax.set_aspect(0.15)
 #   ax.set_aspect("auto")

    p.yticks([0.5, 0.9, 0.99, 1.01, 1.1, 2])

    if filename == None:
        p.show()    
    
    else:
        p.savefig(filename)

def visualizeColorMesh(X, Y, Z, roots, normalizationAngle=None, filename=None):
#    p.pcolormesh(X, Y, Z, cmap=cm.gnuplot2)
    p.clf()

    cdict5 = createCMapDictHelix(10)

    helix = LinearSegmentedColormap("helix", cdict5)

    p.register_cmap(cmap=helix)

    p.pcolormesh(X, Y, Z, cmap=helix)#, vmin=-5, vmax=30)
    
    p.colorbar()
    
    for root in roots:
        if type(root) == type((0,1)):
            actualRoot = root[0]
            color = root[1]

            if type(actualRoot) == type(np.array([0,1])) and len(actualRoot) == 2:

                p.plot(actualRoot[0], actualRoot[1], color + ".")

            else:

                p.plot(np.real(actualRoot), np.imag(actualRoot), color + ".")


        else:
            if type(root) == type(np.array([0,1])) and len(root) == 2:

                p.plot(root[0], root[1], "c.")

            else:

#            p.plot(cis(root), np.abs(root), "w.")
                p.plot(np.real(root), np.imag(root), "c.")                

#    for root in roots:
#        p.plot(np.real(root), np.imag(root), "w.")
    
    unitCircle = p.Circle((0, 0), 1, color='w', fill=False)    

    if normalizationAngle is not None:
        p.plot([0,3*cos(normalizationAngle)], [0, 3*sin(normalizationAngle)], "w-")

    ax = p.gca()
        
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])    
    ax.set_aspect("equal")
    ax.add_artist(unitCircle) 
    
    if filename == None:
        p.show()    
    
    else:
        p.savefig(filename)
 
def truncateFrame(singleColorFrame):
    littleThirdLength = int(len(singleColorFrame - 2)/3)
#    littleThirdLength = 3

    print littleThirdLength

    print [0]*littleThirdLength, singleColorFrame[littleThirdLength:-littleThirdLength], \
        [0]*littleThirdLength

    return np.array([0]*littleThirdLength + [i for i in \
        singleColorFrame[littleThirdLength:-littleThirdLength]] + [0]*littleThirdLength)

def truncate(listOfSingleColorFrames):
    
    listOfTruncatedSingleColorFrames = []

    for singleColorFrame in listOfSingleColorFrames:
        listOfTruncatedSingleColorFrames.append(truncateFrame(singleColorFrame))

    return np.array(listOfTruncatedSingleColorFrames)

def average(l):
    return sum(l)/len(l)

def bwifyFlatFrame(flatFrame):
    returnArray = []

    for rgb in flatFrame:
        returnArray.append(average(rgb))

    return np.array(returnArray)

# function only defined on 0,1
def sequenceToFunction(seq):
    def returnFunc(x):
        if x < 0:
            return None
        if x > 1:
            return None
        if x == 1:
            return seq[-1]
        n = len(seq)

        return seq[int(floor(n*x))]
    return returnFunc

def roadsideConvolve(sceneFunc, occFunc, d):

    xRange = np.linspace(0, 1, 100)
    yRange = np.linspace(d, 1, 70)

#    X, Y = np.meshgrid(xRange, yRange)

    def integrandMaker(xp, yp, d):
        def integrand(x):
            return sceneFunc(x)*occFunc((yp-d)*x/yp + d*xp/yp)

        return integrand

    def findValMaker(d):
        def findVal(xp, yp):
            integrand = integrandMaker(xp, yp, d)

            returnVal = quad(integrand, 0, 1)[0]
#            print returnVal
            return returnVal

        return findVal

    findVal = findValMaker(d)

    returnArray = []

    for i, yp in enumerate(yRange):
        print i, "/", len(yRange)

        returnArray.append([])
        for xp in xRange:
            returnArray[-1].append(findVal(xp, yp))

    print returnArray

    return np.array(returnArray)

def gammaDistMaker(k, theta):
    def gammaDist(x):
        return x**(k-1)*exp(-x/theta)/(theta**k * gamma(k))

    return gammaDist

def absNormalMaker(sigma):
    def absNormal(x):
        return sigma*sigma*x*exp(-x*x/2*sigma*sigma)

    return absNormal

def invAbsNormalMaker(sigma, prodVal):
    absNormal = absNormalMaker(sigma)

    def invAbsNormal(x):
        return prodVal/x**2 * absNormal(prodVal/x)

    return invAbsNormal

def distribProductMaker(listOfDistribs):
    def distribProduct(x):
        prod = 1
        for distrib in listOfDistribs:
            prod *= distrib(x)

        return prod
    return distribProduct

def logDistribProductMaker(listOfDistribs):
    def logDistribProduct(x):
        logSum = 0
        for distrib in listOfDistribs:
            if distrib(x) == 0:
                logSum += -1e4
            else:
                logSum += log(distrib(x))

        return logSum
    return logDistribProduct

def generateGaussianSeq(n, sigma):
    return [np.random.normal(0, sigma) for _ in range(n)]

def wrapConvolve(seq1, seq2):
    return np.fft.ifft(np.multiply(np.fft.fft(seq1), np.fft.fft(seq2)))/sqrt(n)

def wrapDeconvolve(seq1, seq2):
    return np.fft.ifft(np.divide(np.fft.fft(seq1), np.fft.fft(seq2)))/sqrt(n)

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

def getForwardModelMatrix2DToeplitzFull(occ):
    returnMat = []
    occShape = occ.shape

    paddedOcc = padArrayToShape(occ, (3*occShape[0]-2, 3*occShape[1]-2))

    for i in range(2*occShape[0]-1):
        for j in range(2*occShape[1]-1):
            returnMat.append((paddedOcc[i:i+occShape[0], j:j+occShape[1]]).flatten())

    return np.flip(np.array(returnMat), 1)

def getForwardModelMatrix2DToeplitzFullFlexibleShape(occ, obsShape, extra=(0,0), padVal=0):
    returnMat = []
    occShape = occ.shape

    paddedOcc = padArrayToShape(occ, (2*obsShape[0]-occShape[0]+extra[0], \
        2*obsShape[1]-occShape[1]+extra[1]), padVal=padVal)

    viewFrame(imageify(paddedOcc), adaptiveScaling=True)

    print paddedOcc.shape

    for i in range(obsShape[0]):
        for j in range(obsShape[1]):
#            print (paddedOcc[i:i+obsShape[0]-occShape[0], j:j+obsShape[1]-occShape[1]]).shape

            returnMat.append((paddedOcc[i:i+obsShape[0]-occShape[0]+extra[0], j:j+obsShape[1]-occShape[1]+extra[1]]).flatten())

#    print np.array(returnMat).shape

#    print paddedOcc.shape

    return np.flip(np.array(returnMat), 1)

def getPseudoInverse(mat, snr):
    n = mat.shape[1]

    print mat.shape

    thingToBeInverted = snr*np.dot(np.transpose(mat), mat) + np.identity(n)

    print np.dot(np.transpose(mat), mat)[0][0]

    p.matshow(thingToBeInverted)
#    p.matshow(np.log(thingToBeInverted))
    p.colorbar()
    p.show()

    return snr*np.dot(np.linalg.inv(thingToBeInverted), np.transpose(mat))

def getPseudoInverseSmooth(mat, convolvedFrameShape, snr):
    n = mat.shape[0]
    m = mat.shape[1]

    attenMat = getAttenuationMatrix(convolvedFrameShape, 0.1)
#    print attenMat.shape

#    p.matshow(attenMat)
#    p.colorbar()
#    p.show()

    print "computing diag..."

    diagAttenMat = np.dot(np.dot(np.kron(dftMat(convolvedFrameShape[0]), dftMat(convolvedFrameShape[1])), \
        np.diag(attenMat.flatten())), \
        np.kron(np.conj(dftMat(convolvedFrameShape[0])), np.conj(dftMat(convolvedFrameShape[1]))))

#    p.matshow(np.real(diagAttenMat))
#    p.colorbar()
#    p.show()

    print "computing pseudoinverse..."

#    print diagAttenMat.shape

    return (snr+1)*np.dot(np.linalg.inv(snr*np.dot(np.dot(np.transpose(mat), diagAttenMat), mat) + \
        np.identity(m)), np.transpose(mat))

#    return (snr+1)*np.dot(np.transpose(mat), np.linalg.inv(snr*np.dot(np.dot(mat, diagAttenMat), np.transpose(mat)) + \
#        np.identity(n)))


def vectorizedDotToVector(mat, arr):
    return np.dot(mat, arr.flatten())

def vectorizedDot(mat, arr, targetShape):
    return np.reshape(np.dot(mat, arr.flatten()), targetShape)

def getFirstMatch(diffVid, occ, vid):

#    frame = random.choice(diffVid)
    frame = diffVid[30]

    convolvedFrame = np.abs(convolve2DToeplitzFull(frame, occ))

    canvas = convolvedFrame
    canvasShape = canvas.shape

    occShape = occ.shape

    frameCounter = 1

    canvasSimilarities = []

#    random.shuffle(vid)

    while frameCounter < 100:

        print frameCounter

        if frameCounter % 200 == 0:

#            viewFrame(imageify(canvas), adaptiveScaling=True, differenceImage=True)

            extractedOcc = extractOccluderFromCanvas(canvas, occShape)

            viewFrame(imageify(extractedOcc), adaptiveScaling=True, filename="occluder_" + str(frameCounter) + ".png")

            randomFrame = random.choice(vid)

            convolvedRandomFrame = addNoise(doFuncToEachChannel(lambda x: convolve2DToeplitzFull(x, occ), randomFrame))

            forwardModelMatrix = getForwardModelMatrix2DToeplitzFull(extractedOcc)
#            p.matshow(forwardModelMatrix)
#            p.show()

            inversionMatrix = getPseudoInverse(forwardModelMatrix, 1e1)
#            inversionMatrixSmooth = getPseudoInverseSmooth(forwardModelMatrix, randomFrame.shape[:-1], 1e1)            

            recoveredFrame = doFuncToEachChannel(lambda x: vectorizedDot(inversionMatrix, x, randomFrame.shape[:-1]), \
                convolvedRandomFrame)

#            recoveredFrameSmooth = doFuncToEachChannel(lambda x: vectorizedDot(inversionMatrixSmooth, x, randomFrame.shape[:-1]), \
#                convolvedRandomFrame)           


            viewFrame(randomFrame, filename="frame_gt_" + str(frameCounter) + ".png")


            viewFrame(recoveredFrame, adaptiveScaling=True, magnification=1, filename="frame_recovery_" \
                + str(frameCounter) + "_mag1.png")

            viewFrame(recoveredFrame, adaptiveScaling=True, magnification=3, filename="frame_recovery_" \
                + str(frameCounter) + "_mag3.png")

            viewFrame(recoveredFrame, adaptiveScaling=True, magnification=10, filename="frame_recovery_" \
                + str(frameCounter) + "_mag10.png")



            for i, candidateOcc in enumerate(convertOccToZeroOne(extractedOcc)):

                fractionOfOnes = np.sum(candidateOcc)/np.size(candidateOcc)

                if fractionOfOnes > 0.25 and fractionOfOnes < 0.75:

                    viewFrame(imageify(candidateOcc), adaptiveScaling=True, filename="occluder_" + str(frameCounter) + "_" + \
                        str(i) + ".png")

                    print "computing forward model"

                    forwardModelMatrix = getForwardModelMatrix2DToeplitzFull(candidateOcc)
#                    p.matshow(forwardModelMatrix)
 #                   p.show()
                    print "computing pseudoinverse"


                    inversionMatrix = getPseudoInverse(forwardModelMatrix, 1e1)
        #            inversionMatrixSmooth = getPseudoInverseSmooth(forwardModelMatrix, randomFrame.shape[:-1], 1e1)            

                    print "doing recovery"




#                    diffFrame = random.choice(diffVid)

#                    recoveredDiffFrame = vectorizedDot(inversionMatrix, convolvedDiffFrame, diffFrame.shape)

         


                    recoveredFrame = doFuncToEachChannel(lambda x: vectorizedDot(inversionMatrix, x, randomFrame.shape[:-1]), \
                        convolvedRandomFrame)

        #            recoveredFrameSmooth = doFuncToEachChannel(lambda x: vectorizedDot(inversionMatrixSmooth, x, randomFrame.shape[:-1]), \
        #                convolvedRandomFrame)                

                    viewFrame(recoveredFrame, adaptiveScaling=True, magnification=1, filename="frame_recovery_" \
                        + str(frameCounter) + "_" + str(i) + "_mag1.png")

                    viewFrame(recoveredFrame, adaptiveScaling=True, magnification=3, filename="frame_recovery_" \
                        + str(frameCounter) + "_" + str(i) + "_mag3.png")

                    viewFrame(recoveredFrame, adaptiveScaling=True, magnification=10, filename="frame_recovery_" \
                        + str(frameCounter) + "_" + str(i) + "_mag10.png")
        

#            viewFrame(recoveredFrame, adaptiveScaling=True, magnification=1, filename="frame_recovery_" \
 #               + str(frameCounter) + "_smooth_mag1.png")

#            viewFrame(recoveredFrame, adaptiveScaling=True, magnification=3, filename="frame_recovery_" \
 #               + str(frameCounter) + "_smooth_mag3.png")

#            viewFrame(recoveredFrame, adaptiveScaling=True, magnification=10, filename="frame_recovery_" \
 #               + str(frameCounter) + "_smooth_mag10.png")

#        frame = random.choice(diffVid)
        frame = diffVid[30 + frameCounter]

#        frame = random.choice(vid)
#        frame = np.swapaxes(np.swapaxes(frame, 1, 2), 0, 1)[random.randint(0, 2)]

#        print frame
#        print convolvedFrame

        convolvedFrame = np.abs(addNoise(convolve2DToeplitzFull(frame, occ)))

#        viewFrame(imageify(convolvedFrame), adaptiveScaling=True, differenceImage=False)
 #       viewFrame(imageify(canvas), adaptiveScaling=True, differenceImage=False)


        matchArray, bestMatchArray, bestMatchIndex, matchQuality = \
            getMatchArray(canvas, convolvedFrame)

#        viewFrame(imageify(matchArray), adaptiveScaling=True)

        overlapArray1, overlapArray2 = getOverlapArray(canvas, \
            convolvedFrame, bestMatchIndex)

#        viewFrame(imageify(overlapArray1), adaptiveScaling=True)
 #       viewFrame(imageify(overlapArray2), adaptiveScaling=True)

        if np.shape(overlapArray1)[0] > 30 and np.shape(overlapArray1)[1] > 30 and \
            np.shape(overlapArray2)[0] > 30 and np.shape(overlapArray2)[1] > 30:

            overallOverlapArray = np.multiply(np.power(overlapArray1, frameCounter/(frameCounter+1)), \
                np.power(overlapArray2, 1/(frameCounter+1)))

#            viewFrame(imageify(overallOverlapArray), adaptiveScaling=True)

            canvas = padArrayToShape(overallOverlapArray, canvasShape)

#            viewFrame(imageify(canvas), adaptiveScaling=True)

#        viewFrame(imageify(canvas), differenceImage=False, adaptiveScaling=True)

        extractedOcc = extractOccluderFromCanvas(canvas, occShape)

        canvasSimilarity = np.sum(np.multiply(extractedOcc, occ))/np.sum(np.sqrt(extractedOcc))/np.sum(np.sqrt(occ))
        canvasSimilarities.append(canvasSimilarity)

#        viewFrame(imageify(extractedOcc), adaptiveScaling=True, filename="canvas_video/extracted_occ_" + \
#            padIntegerWithZeros(frameCounter, 3) + ".png")
#        viewFrame(imageify(convolvedFrame), adaptiveScaling=True, differenceImage=True, filename="canvas_video/conv_frame_" + \
#            padIntegerWithZeros(frameCounter, 3) + ".png")

        frameCounter += 1

    viewFrame(imageify(extractedOcc), adaptiveScaling=True)
    pickle.dump(extractedOcc, open("extracted_occ.p", "w"))

#    p.clf()
#    p.plot(canvasSimilarities)
 #   p.savefig("canvas_video/canvas_similarities.png")

def estimateOccluderFromDifferenceFrames(diffVid, canvasShape):
    random.shuffle(diffVid)
    convolvedFrame = random.choice(diffVid)

    canvas = np.abs(convolvedFrame)

    frameCounter = 1
    viewFrame(imageify(canvas), adaptiveScaling=True)

    for convolvedFrame in diffVid[1:]:
        absConvolvedFrame = np.abs(convolvedFrame)

        if np.sum(convolvedFrame) > 0:

            matchArray, bestMatchIndex, = \
                getMatchArrayUnequalSize(canvas, absConvolvedFrame)               

            overlapArray1, overlapArray2 = getOverlapArrayPadded(canvas, absConvolvedFrame, \
                bestMatchIndex)

#            canvas = overlapArray1 + overlapArray2
            canvas = np.multiply(overlapArray1, overlapArray2)

            canvas = cutArrayDownToShape(canvas, canvasShape)

            viewFrame(imageify(canvas), adaptiveScaling=True)

def estimateOccluderFromDifferenceFramesOld(diffVid):

    random.shuffle(diffVid)

    convolvedFrame = random.choice(diffVid)

    canvas = np.abs(convolvedFrame)
    canvasShape = canvas.shape

    print canvasShape

    frameCounter = 1
#    viewFrame(imageify(canvas), adaptiveScaling=True)

    for convolvedFrame in diffVid[6:]:

#        viewFrame(imageify(convolvedFrame), adaptiveScaling=True)

        print frameCounter

        absConvolvedFrame = np.abs(convolvedFrame)

        if frameCounter % 10 == 0:
            viewFrame(imageify(np.log(canvas + np.ones(canvasShape))), adaptiveScaling=True)
            viewFrame(imageify(canvas), adaptiveScaling=True)

        if np.sum(convolvedFrame) > 0:

            matchArray, bestMatchArray, bestMatchIndex, matchQuality = \
                getMatchArray(canvas, absConvolvedFrame)

    #        viewFrame(imageify(matchArray), adaptiveScaling=True)

            overlapArray1, overlapArray2 = getOverlapArray(canvas, \
                absConvolvedFrame, bestMatchIndex)

#            viewFrame(imageify(overlapArray1), adaptiveScaling=True, differenceImage=True)
#            viewFrame(imageify(overlapArray2), adaptiveScaling=True, differenceImage=True)



            if np.shape(overlapArray1)[0] > canvasShape[0]*0.75 and \
                np.shape(overlapArray1)[1] > canvasShape[1]*0.75 and \
                np.shape(overlapArray2)[0] > canvasShape[0]*0.75 and \
                np.shape(overlapArray2)[1] > canvasShape[1]*0.75:

                overallOverlapArray = np.multiply(np.power(overlapArray1, frameCounter/(frameCounter+1)), \
                    np.power(overlapArray2, 1/(frameCounter+1)))

                overlapArray1Shape = overlapArray1.shape
                overlapArray2Shape = overlapArray2.shape

                overallOverlapArray = np.exp(((frameCounter-1)/frameCounter)*np.log(overlapArray1 + np.ones(overlapArray1Shape)) \
                                    + (1/frameCounter)*np.log(overlapArray2 + np.ones(overlapArray2Shape)))



#                viewFrame(imageify(overallOverlapArray), adaptiveScaling=True)



                canvas = padArrayToShape(overallOverlapArray, canvasShape)

#                viewFrame(imageify(convolvedFrame), adaptiveScaling=True)
#                viewFrame(imageify(canvas), adaptiveScaling=True)

    #        viewFrame(imageify(canvas), differenceImage=False, adaptiveScaling=True)

    #        viewFrame(imageify(extractedOcc), adaptiveScaling=True, filename="canvas_video/extracted_occ_" + \
    #            padIntegerWithZeros(frameCounter, 3) + ".png")
    #        viewFrame(imageify(convolvedFrame), adaptiveScaling=True, differenceImage=True, filename="canvas_video/conv_frame_" + \
    #            padIntegerWithZeros(frameCounter, 3) + ".png")

            frameCounter += 1

    pickle.dump(canvas, open("extracted_occ_exp.p", "w"))    

    return canvas

def estimateOccluderFromDifferenceFramesCanvasPreserving(diffVid):

#    random.shuffle(diffVid)
    
#    start = 420
    start = 1172

    convolvedFrame = diffVid[start]

#    convolvedFrame = random.choice(diffVid)
#    convolvedFrame = 1e-5*np.ones(diffVid[0].shape)

    print len(diffVid)

#    canvas = np.abs(convolvedFrame)
    canvas = convolvedFrame

    viewFrame(imageify(canvas), adaptiveScaling=True)
    canvasShape = canvas.shape

#    pickle.dump(canvas, open("fan_extracted_occ.p", "w"))    

    print canvasShape

    frameCounter = start
#    viewFrame(imageify(canvas), adaptiveScaling=True)

    contribCounter = 1

    for i, convolvedFrame in enumerate(diffVid[start:]):

        convolvedFrame = diffVid[frameCounter]

#        viewFrame(imageify(convolvedFrame), adaptiveScaling=True)

#        print i

#        print frameCounter

        absConvolvedFrame = np.abs(convolvedFrame)

        if frameCounter % 1 == 0:
#            viewFrame(imageify(np.log(canvas + np.ones(canvasShape))), adaptiveScaling=True)
            viewFrame(imageify(canvas), adaptiveScaling=True)

#            for recoveredOcc in convertOccToZeroOne(canvas):
#                viewFrame(imageify(recoveredOcc))
            pass

        if np.sum(absConvolvedFrame) > 3000:

            matchArray, bestMatchArray, bestMatchIndex, matchQuality = \
                getMatchArray(canvas, absConvolvedFrame)

#            viewFrame(imageify(absConvolvedFrame), adaptiveScaling=True)

    #        viewFrame(imageify(matchArray), adaptiveScaling=True)

            overlapArray = getOverlapArrayFullCanvas(canvas, \
                absConvolvedFrame, bestMatchIndex)

#            viewFrame(imageify(overlapArray1), adaptiveScaling=True, differenceImage=True)
#            viewFrame(imageify(overlapArray2), adaptiveScaling=True, differenceImage=True)

#            if np.shape(overlapArray1)[0] > canvasShape[0]*0.75 and \
#                np.shape(overlapArray1)[1] > canvasShape[1]*0.75 and \
#                np.shape(overlapArray2)[0] > canvasShape[0]*0.75 and \
#                np.shape(overlapArray2)[1] > canvasShape[1]*0.75:

#            overallOverlapArray = np.multiply(np.power(canvas, frameCounter/(frameCounter+1)), \
#                np.power(overlapArray, 1/(frameCounter+1)))

            c = contribCounter

            overallOverlapArray = canvas*(c/(c+1)) + \
                overlapArray* (1/(c+1))

#            overallOverlapArray = np.exp(((frameCounter-1)/frameCounter)*np.log(canvas + np.ones(canvasShape)) \
#                                + (1/frameCounter)*np.log(overlapArray + np.ones(canvasShape)))

            canvas = padArrayToShape(overallOverlapArray, canvasShape)

#                viewFrame(imageify(convolvedFrame), adaptiveScaling=True)
#                viewFrame(imageify(canvas), adaptiveScaling=True)

    #        viewFrame(imageify(canvas), differenceImage=False, adaptiveScaling=True)

    #        viewFrame(imageify(extractedOcc), adaptiveScaling=True, filename="canvas_video/extracted_occ_" + \
    #            padIntegerWithZeros(frameCounter, 3) + ".png")
    #        viewFrame(imageify(convolvedFrame), adaptiveScaling=True, differenceImage=True, filename="canvas_video/conv_frame_" + \
    #            padIntegerWithZeros(frameCounter, 3) + ".png")

            contribCounter += 1

        frameCounter += 3

        print frameCounter, start+20

        if frameCounter > start + 20:
            break

    pickle.dump(canvas, open("plant_extracted_occ.p", "w"))    

    return canvas

def estimateSceneFromDifferenceFramesCanvasPreserving(diffVid):
    convolvedFrame = random.choice(diffVid)

    start

    diffVidShuffled = diffVid.copy()
    random.shuffle(diffVidShuffled)

#    print len(diffVid)

    canvasDims = convolvedFrame.shape[:-1]
    canvas = None

    frameCounter = 1

    diffVidSorted = sorted(diffVid, key=lambda x: -np.sum(np.abs(x)))

    overlapArray = np.zeros(convolvedFrame.shape)

    for i, convolvedFrame in enumerate(diffVidSorted[1500:]):
#        convolvedFrame = random.choice(diffVidShuffled)

        if i % 100 == 0 and i > 0:
            print "canvas"
            viewFrame(canvas, adaptiveScaling=True, differenceImage=True)

            print "most recent frame"
            viewFrame(convolvedFrame, adaptiveScaling=True, differenceImage=True)

#            print "most recent overlap array"
#            viewFrame(overlapArray, adaptiveScaling=True, differenceImage=True)
        if canvas == None:
#            canvas = convolvedFrame
            canvas = np.abs(convolvedFrame)

        else:

            convolvedFrameStrength = np.sum(np.abs(convolvedFrame))
            canvasStrength = np.sum(np.abs(canvas))
            
            if convolvedFrameStrength > 200:

                matchArray, bestIndex = getMatchArrayRGB(canvas, convolvedFrame)

    #            listOfMatchResults = doFuncToEachChannelSeparated(lambda x: getMatchArray(x, convolvedFrame), canvas)
                
#                print "match array"
#                viewFrame(imageify(matchArray), adaptiveScaling=True)

#                overlapArray = doFuncToEachChannelTwoInputs(lambda x, y: getOverlapArrayFullCanvas(x, y, bestIndex), \
#                    canvas, convolvedFrame)

                overlapArray = doFuncToEachChannelTwoInputs(lambda x, y: getOverlapArrayFullCanvas(x, y, bestIndex), \
                    canvas, np.abs(convolvedFrame))


    #            print "overlap array"
    #            viewFrame(overlapArray, adaptiveScaling=True)
    #            viewFrame(imageify(overlapArray1), adaptiveScaling=True, differenceImage=True)
    #            viewFrame(imageify(overlapArray2), adaptiveScaling=True, differenceImage=True)

    #            if np.shape(overlapArray1)[0] > canvasShape[0]*0.75 and \
    #                np.shape(overlapArray1)[1] > canvasShape[1]*0.75 and \
    #                np.shape(overlapArray2)[0] > canvasShape[0]*0.75 and \
    #                np.shape(overlapArray2)[1] > canvasShape[1]*0.75:

    #            overallOverlapArray = np.multiply(np.power(canvas, frameCounter/(frameCounter+1)), \
    #                np.power(overlapArray, 1/(frameCounter+1)))

                print convolvedFrameStrength

                overallOverlapArray = canvas/canvasStrength * (frameCounter/(frameCounter+1)) + \
                    overlapArray/convolvedFrameStrength * (1/frameCounter)

#                overallOverlapArray = canvas *(frameCounter/(frameCounter+1)) + \
#                    overlapArray*(1/frameCounter)

    #            overallOverlapArray = np.exp(((frameCounter-1)/frameCounter)*np.log(canvas + np.ones(canvasShape)) \
    #                                + (1/frameCounter)*np.log(overlapArray + np.ones(canvasShape)))

                canvas = doFuncToEachChannel(lambda x: padArrayToShape(x, canvasDims), \
                    overallOverlapArray)

                frameCounter += 1

            if frameCounter > 300:
                break

    pickle.dump(canvas, open("bld34_scene_exp.p", "w"))    

    return canvas


def convertOccToZeroOne(occ):

    averageVal = np.sum(occ)/np.size(occ)

    candidateOccs = []

    for logOffset in np.linspace(-0.5, 0.5, 50):

        offset = 10**logOffset

        candidateOccs.append(np.vectorize(lambda x: 1.0*(x > offset*averageVal))(occ))

    return candidateOccs

def cutArrayDownToShape(arr, shape, reverse=False):
    if reverse:
        convolvedWithWhite = convolve2DToeplitz(-np.ones(shape), arr)
    else:    
        convolvedWithWhite = convolve2DToeplitz(np.ones(shape), arr)

    bestIndex = np.unravel_index(np.argmax(convolvedWithWhite, axis=None), convolvedWithWhite.shape)

    return arr[bestIndex[0]:bestIndex[0]+shape[0], bestIndex[1]:bestIndex[1]+shape[1]]

def cutArrayDownToShapeWithAlternateArray(arr, altArr, shape, reverse=False):
    if reverse:
        convolvedWithWhite = convolve2DToeplitz(-np.ones(shape), altArr)
    else:    
        convolvedWithWhite = convolve2DToeplitz(np.ones(shape), altArr)

    bestIndex = np.unravel_index(np.argmax(convolvedWithWhite, axis=None), convolvedWithWhite.shape)

    return arr[bestIndex[0]:bestIndex[0]+shape[0], bestIndex[1]:bestIndex[1]+shape[1]]

def extractOccluderFromCanvas(canvas, occShape):
    convolvedWithWhite = convolve2DToeplitz(np.ones(occShape), canvas)

    bestIndex = np.unravel_index(np.argmax(convolvedWithWhite, axis=None), convolvedWithWhite.shape)

    return canvas[bestIndex[0]:bestIndex[0]+occShape[0], bestIndex[1]:bestIndex[1]+occShape[1]]




#def deconvolve

#        viewFrame(imageify(overlapArray1), adaptiveScaling=True)
 #       viewFrame(imageify(overlapArray2), adaptiveScaling=True)
        
 #       viewFrame(imageify(np.multiply(overlapArray1, overlapArray2)), adaptiveScaling=True)

 #       if matchQuality > 20000 and overlapArray1.shape[0] > 30 and overlapArray2.shape[1] > 30:
 #           break

if __name__ == "__main__":

    if LOOK_AT_FREQUENCY_PROFILES:

        n = 100

        sparseSeq = generateSparseSeq(n)
        zeroOneSeq = generateZeroOneSeq(n)

        #sparseSeq = generateImpulseSeq(n)
        #zeroOneSeq = generateImpulseSeq(n)




        viewFlatFrame(imageify(sparseSeq))
        viewFlatFrame(imageifyComplex(fft(sparseSeq)), differenceImage=True, magnification=0.05)
        viewFlatFrame(imageify(np.abs(fft(sparseSeq))), differenceImage=False, magnification=0.05)
        viewFlatFrame(imageifyComplex(fft(sparseSeq)/np.abs(fft(sparseSeq))), differenceImage=True, magnification=1)
        viewFlatFrame(imageify(zeroOneSeq))
        viewFlatFrame(imageifyComplex(fft(zeroOneSeq)), differenceImage=True, magnification=0.05)
        viewFlatFrame(imageify(np.abs(fft(zeroOneSeq))), differenceImage=False, magnification=0.05)
        viewFlatFrame(imageifyComplex(fft(zeroOneSeq)/np.abs(fft(zeroOneSeq))), differenceImage=True, magnification=1)


        convSeq = conv(sparseSeq, zeroOneSeq)
        noisyConvSeq = addNoise(convSeq)

        viewFlatFrame(imageify(noisyConvSeq))
        viewFlatFrame(imageifyComplex(fft(noisyConvSeq)), differenceImage=True, magnification=0.0025)
        viewFlatFrame(imageify(np.abs(fft(noisyConvSeq))), differenceImage=False, magnification=0.0025)
        viewFlatFrame(imageifyComplex(fft(noisyConvSeq)/np.abs(fft(noisyConvSeq))), differenceImage=True, magnification=1)

        deconv(noisyConvSeq)

    if DIVIDE_OUT_STRATEGY:
        n = 100

        sparseSeq1 = generateSparseSeq(n)
        sparseSeq2 = generateSparseSeq(n)
        sparseSeq3 = generateSparseSeq(n)
        zeroOneSeq = generateZeroOneSeq(n)
        
        sparseSeqs = [generateSparseSeq(n) for _ in range(10)]
        
    #    viewFlatFrame(imageify(sparseSeq1), differenceImage=True)
    #    viewFlatFrame(imageify(sparseSeq2), differenceImage=True)
    #    viewFlatFrame(imageify(sparseSeq3), differenceImage=True)
    #    viewFlatFrame(imageify(zeroOneSeq))    
        
        viewFlatFrame(imageifyComplex(fft(zeroOneSeq)), differenceImage=True, magnification=0.05)

        viewFlatFrame(imageifyComplex(fft(sparseSeq1)), differenceImage=True, magnification=0.05)
        viewFlatFrame(imageifyComplex(fft(sparseSeq2)), differenceImage=True, magnification=0.05)
        viewFlatFrame(imageifyComplex(fft(sparseSeq3)), differenceImage=True, magnification=0.05)        
            
        noisyConvSeq1 = addNoise(conv(sparseSeq1, zeroOneSeq))
        noisyConvSeq2 = addNoise(conv(sparseSeq2, zeroOneSeq))
        noisyConvSeq3 = addNoise(conv(sparseSeq3, zeroOneSeq))
        
        noisyConvSeqs = [addNoise(conv(sparseSeq, zeroOneSeq)) for sparseSeq in sparseSeqs]

        viewFlatFrame(imageifyComplex(fft(noisyConvSeq1)), differenceImage=True, magnification=0.05)
        viewFlatFrame(imageifyComplex(fft(noisyConvSeq2)), differenceImage=True, magnification=0.05)
        viewFlatFrame(imageifyComplex(fft(noisyConvSeq3)), differenceImage=True, magnification=0.05)
            
        viewFlatFrame(imageify(sparseSeq1), differenceImage=True)
         
        for i, noisyConvSeq in enumerate(noisyConvSeqs):
            viewFlatFrame(imageify(sparseSeqs[i]), differenceImage=True)
            viewFlatFrame(imageify(ifft(numpyDivideNoDivZero(fft(noisyConvSeq1), fft(noisyConvSeq)))), \
                differenceImage=True) 
            viewFlatFrame(imageify(ifft(numpyDivideNoDivZero(fft(noisyConvSeq), fft(noisyConvSeq1)))), \
                differenceImage=True)      
            
        divSeq12 = numpyDivideNoDivZero(fft(noisyConvSeq1), fft(noisyConvSeq2))    
        divSeq13 = numpyDivideNoDivZero(fft(noisyConvSeq1), fft(noisyConvSeq3))    
        divSeq21 = numpyDivideNoDivZero(fft(noisyConvSeq2), fft(noisyConvSeq1))   
        divSeq23 = numpyDivideNoDivZero(fft(noisyConvSeq2), fft(noisyConvSeq3))    
        divSeq31 = numpyDivideNoDivZero(fft(noisyConvSeq3), fft(noisyConvSeq1))   
        divSeq32 = numpyDivideNoDivZero(fft(noisyConvSeq3), fft(noisyConvSeq2))    
            
        
        
        
        
        
        viewFlatFrame(imageifyComplex(divSeq12), differenceImage=True, magnification=0.5)
        viewFlatFrame(imageifyComplex(divSeq13), differenceImage=True, magnification=0.5)
        viewFlatFrame(imageifyComplex(divSeq23), differenceImage=True, magnification=0.5)

        viewFlatFrame(imageify(np.abs(divSeq12)), differenceImage=False, magnification=0.5)
        viewFlatFrame(imageify(np.abs(divSeq13)), differenceImage=False, magnification=0.5)
        viewFlatFrame(imageify(np.abs(divSeq23)), differenceImage=False, magnification=0.5)

        viewFlatFrame(imageifyComplex(divSeq12/np.abs(divSeq12)), differenceImage=True, magnification=1)
        viewFlatFrame(imageifyComplex(divSeq13/np.abs(divSeq13)), differenceImage=True, magnification=1)
        viewFlatFrame(imageifyComplex(divSeq23/np.abs(divSeq23)), differenceImage=True, magnification=1)

        viewFlatFrame(imageify(ifft(divSeq12)), differenceImage=True)
        viewFlatFrame(imageify(ifft(divSeq13)), differenceImage=True)
        viewFlatFrame(imageify(ifft(divSeq21)), differenceImage=True)
        viewFlatFrame(imageify(ifft(divSeq23)), differenceImage=True)
        viewFlatFrame(imageify(ifft(divSeq31)), differenceImage=True)
        viewFlatFrame(imageify(ifft(divSeq32)), differenceImage=True)    
        
    if POLYNOMIAL_STRATEGY:
        listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)

        n = len(listOfFlatFrames[0])

        pront("n = " + n)

        listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
            for i in range(len(listOfFlatFrames) - 1)]

        occluder = generateZeroOneSeq(2*n-1)
            
        viewFlatFrame(imageify(occluder))

        occluderPolynomial = Polynomial(occluder)

        displayRoots(occluderPolynomial, [], 0, 1)

        listOfConvolvedDifferenceFrames = [addNoise(doFuncToEachChannelVec(convolveMaker(occluder), frame)) for frame in listOfFlatDifferenceFrames]

        concatenatedDifferenceFrames = np.concatenate(listOfFlatDifferenceFrames, 1)
        convolvedDifferenceFrames = np.concatenate(listOfConvolvedDifferenceFrames, 1)
        
        viewFrame(concatenatedOriginalFrames, magnification=1, differenceImage=True)
        viewFrame(concatenatedDifferenceFrames, magnification=100, differenceImage=True)
        
        
    #    displayConcatenatedArray(concatenatedDifferenceFrames, magnification=100, \
    #        differenceImage=True, rowsPerFrame=1)
        
    #    displayConcatenatedArray(convolvedDifferenceFrames, magnification=100, \
    #        differenceImage=True, rowsPerFrame=1)    
        
        listOfPolynomials = getListOfPolynomialsFromConvolvedDifferenceFrames(listOfConvolvedDifferenceFrames)
        listOfDistractionPolynomials = getListOfPolynomialsFromConvolvedDifferenceFrames(listOfFlatDifferenceFrames)
        
      #  random.shuffle(listOfPolynomials)
        
        rootDict = {}
        
        listOfExistingRoots = [[root, 1] for root in listOfPolynomials[0].roots()]
        
        blueThresh = 0.5
        
        
        
        for i, polynomial in enumerate(listOfPolynomials[:10]):
            index = random.randint(0, len(listOfPolynomials)-1)
            polynomial = listOfPolynomials[index]

     #       binaryList = [1*(random.random()<0.5) for _ in range(len(listOfPolynomials))]
     #       polynomial = averageOverPolysInBinList(listOfPolynomials, binaryList)

            viewFlatFrame(imageify(polynomial.coef), magnification=1, differenceImage=True)

    #        distractionPolynomial = listOfDistractionPolynomials[index]
            
            augmentListOfExistingRoots(listOfExistingRoots, polynomial)
            
            displayRoots(Polynomial(ryyNew(polynomial.coef)), [], 0, 1)
            
      #      viewFlatFrame(imageify(polynomial.coef), magnification=1, differenceImage=True)

      #      viewFlatFrame(imageify(distractionPolynomial.coef), magnification=1, differenceImage=True)
            
      #      viewFlatFrame(imageify(occluderPolynomial.coef), magnification=1, differenceImage=True)
            
            
            
      #      displayRoots(polynomial, [], 0, 1)
      #      displayRoots(distractionPolynomial, [], 0, 1)
      #      displayRoots(occluderPolynomial, [], 0, 1)

            listOfRepeatedRoots = displayRoots(polynomial, listOfExistingRoots, blueThresh, i+2)
            
        listOfRepeatedRoots = displayRoots(polynomial, listOfExistingRoots, blueThresh, i+2)    
            
        fitPoly = polyfromroots(listOfRepeatedRoots)
        
        viewFlatFrame(imageify(fitPoly), differenceImage=False)
            
    if UNDO_TRUNCATION:
        listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)

        n = len(listOfFlatFrames[0])

        pront("n = " + str(n))

        listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
    #        for i in range(len(listOfFlatFrames) - 1)]
            for i in range(250,350)]


        occluder = generateZeroOneSeq(2*n-1)
            
        viewFlatFrame(imageify(occluder))
    #    occluderPolynomial = Polynomial(occluder)
    #    displayRoots(occluderPolynomial, [], 0, 1)

        listOfConvolvedDifferenceFrames = [addNoise(doFuncToEachChannelVec(convolveMaker(occluder), frame)) for frame in listOfFlatDifferenceFrames]

        assert len(listOfConvolvedDifferenceFrames[0]) == 3*n-2

        listOfTruncatedDifferenceFrames = np.array([[1*((i >= n-1) and (i < 2*n-1)) * val for i,val in enumerate(cdf)] \
            for cdf in listOfConvolvedDifferenceFrames])

        onVals = np.array([1*((i >= n-1) and (i < 2*n-1)) for i in range(3*n-2)])

        concatenatedDifferenceFrames = np.concatenate(listOfFlatDifferenceFrames, 1)
        convolvedDifferenceFrames = np.concatenate(listOfConvolvedDifferenceFrames, 1)
        truncatedDifferenceFrames = np.concatenate(listOfTruncatedDifferenceFrames, 1)
        
        gamma = 1e3
        onValsFreqMat = circ(fft(onVals))
        p.matshow(np.real(onValsFreqMat))
        p.colorbar()
        p.show()
        
        onValsFreqMatPseudoInverse = gamma*np.dot(np.linalg.inv(gamma*np.dot(np.conj(np.transpose(onValsFreqMat)), \
            onValsFreqMat) + 1*np.identity(onValsFreqMat.shape[0])), np.conj(np.transpose(onValsFreqMat)))
     
        
     
        p.matshow(np.real(gamma*np.dot(np.conj(np.transpose(onValsFreqMat)), \
            onValsFreqMat) + 1*np.identity(onValsFreqMat.shape[0])))
        p.colorbar()
        p.show()
     
        p.matshow(np.linalg.inv(np.real(gamma*np.dot(np.conj(np.transpose(onValsFreqMat)), \
            onValsFreqMat) + 1*np.identity(onValsFreqMat.shape[0]))))
        p.colorbar()
        p.show()
      
        p.matshow(np.real(gamma*np.dot(np.linalg.inv(gamma*np.dot(np.conj(np.transpose(onValsFreqMat)), \
                onValsFreqMat) + 1*np.identity(onValsFreqMat.shape[0])), np.conj(np.transpose(onValsFreqMat)))))
        p.colorbar()
                
        p.show()
      
    #    onValsFreqMatPseudoInverse = np.linalg.inv(onValsFreqMat)
        p.matshow(np.real(onValsFreqMatPseudoInverse))
        p.colorbar()
        p.show()    
            
        displayConcatenatedArray(concatenatedDifferenceFrames, magnification=100, \
            differenceImage=True, rowsPerFrame=1)
        
        displayConcatenatedArray(convolvedDifferenceFrames, magnification=100, \
            differenceImage=True, rowsPerFrame=1)    

        displayConcatenatedArray(truncatedDifferenceFrames, magnification=100, \
            differenceImage=True, rowsPerFrame=1)   
        
        frameNum = 80
        
        tdf = getSingleColorFrames(listOfTruncatedDifferenceFrames[frameNum])[0]
            
        viewFlatFrame(imageify(tdf), differenceImage=True, magnification=1)
        viewFlatFrame(imageifyComplex(fft(tdf)), differenceImage=True, magnification=1)
        viewFlatFrame(imageify(onVals))
        viewFlatFrame(imageifyComplex(fft(onVals)), differenceImage=True, magnification=1)


        viewFlatFrame(imageifyComplex(np.dot(onValsFreqMatPseudoInverse, fft(tdf))), magnification=1)
        viewFlatFrame(imageify(np.real(ifft(np.dot(onValsFreqMatPseudoInverse, fft(tdf))))),
             magnification=1e2, differenceImage=True)

    #    viewFlatFrame(imageifyComplex(deconvolve(fft(tdf), fft(onVals))[0]))
     #   viewFlatFrame(imageifyComplex(ifft(deconvolve(fft(tdf), fft(onVals))[0])))
        
    if BILL_NOISY_POLYNOMIALS:
            
    #    listOfFlatFrames = batchList(pickle.load(open("flat_frames_fine.p", "r")), 2)
    #    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)
    #    listOfFlatFrames = batchList(pickle.load(open("flat_frames_coarse.p", "r")), 2)
        listOfFlatFrames = batchList(pickle.load(open("flat_frames_grey_bar.p")), 1)

        n = len(listOfFlatFrames[0])

        pront("n = " + str(n))

        listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
            for i in range(len(listOfFlatFrames) - 1)]

        print listOfFlatFrames[0]    

    #    occluder = generateZeroOneSeq(2*n-1)    
        occluder = np.array([1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1])
                
    #    occluder = np.array([1,0,0,1,1])

    #    print polyfromroots(np.roots(occluder))

        print np.roots(occluder)

        viewFlatFrame(imageify(occluder))

        occluderPolynomial = Polynomial(occluder[::-1])

    #    displayRoots(occluderPolynomial, [], 0, 1)

        listOfConvolvedDifferenceFrames = [addNoise(doFuncToEachChannelVec(convolveMaker(occluder), frame)) \
            for frame in listOfFlatDifferenceFrames]

        listOfConvolvedFrames = [addNoise(doFuncToEachChannelVec(convolveMaker(occluder), frame)) \
            for frame in listOfFlatFrames] 

        concatenatedOriginalFrames = np.swapaxes(np.concatenate(imageify(np.repeat(np.array(listOfFlatFrames), 1, axis=1))/255, 1), 0, 1)
        concatenatedDifferenceFrames = np.swapaxes(np.concatenate(imageify(np.repeat(np.array(listOfFlatDifferenceFrames), 1, axis=1))/255, 1), 0,1)
        concatenatedConvolvedDifferenceFrames = np.swapaxes(np.concatenate(imageify(np.repeat(np.array(listOfConvolvedDifferenceFrames), 
            1, axis=1))/255, 1), 0,1)
        concatenatedConvolvedFrames = np.swapaxes(np.concatenate(imageify(np.repeat(np.array(listOfConvolvedFrames), 
            1, axis=1))/255, 1), 0,1)

        viewFrame(concatenatedOriginalFrames, magnification=1, differenceImage=False)
        viewFrame(concatenatedDifferenceFrames, magnification=100, differenceImage=True)
        viewFrame(concatenatedConvolvedDifferenceFrames, magnification=100, differenceImage=True)
        viewFrame(concatenatedConvolvedFrames, magnification=1e-1, differenceImage=False)
        
    #    displayConcatenatedArray(concatenatedDifferenceFrames, magnification=100, \
     #       differenceImage=True, rowsPerFrame=10)
        
    #    displayConcatenatedArray(convolvedDifferenceFrames, magnification=100, \
    #        differenceImage=True, rowsPerFrame=10, stretchFactor=1)

        solveProblem = True
        singlePoly = False
        truncation = False
        movieRecovery = False

        densitySigma = 0.3/sqrt(n)
      
        densityFunc = lambda x: logLaplace(x, 0, densitySigma) 

        

        if solveProblem: 

            listOfSingleColorFrames = getListOfSingleColorFrames(listOfConvolvedDifferenceFrames)

            print len(listOfSingleColorFrames)

            listOfRandomlyChosenSingleColorFrames = random.sample(listOfSingleColorFrames, 90)

    #        makeRootMagnitudeHistogram(listOfSingleColorFrames, 0.3/sqrt(n))

    #        rootsFound = findRoots(listOfSingleColorFrames, np.roots(occluder), showy=True)

            rootsFound = findRoots(listOfRandomlyChosenSingleColorFrames, np.roots(occluder), \
                numRootsExpected=len(occluder)-1, showy=False)
     
            print rootsFound

    #        print "correct roots", np.roots(occluder)
     #       print "recovered roots", complexFormRootsFound
    #        print "correct polynomial", polyfromroots(np.roots(occluder))
    #        print "recovered polynomial", polyfromroots(complexFormRootsFound)

    #        print rootsFound

    #        complexFormRootsFound = [np.exp(1j*root[0])*root[1] for root in rootsFound]

            fitPoly = polyfromroots(rootsFound)
            
            pickle.dump(fitPoly, open("fit_poly.p", "w"))

            viewFlatFrame(imageify(fitPoly), differenceImage=True, filename="recovered_occluder.png")

    #        THETA, R, Z = makeAggregateRadialMesh(listOfSingleColorFrames, densityFunc)
            
    #        visualizeRadialColorMesh(THETA, R, Z, densitySigma, np.roots(occluder))
    #        visualizeRadialColorMesh(THETA, R, Z, densitySigma, [])
            
    #        X, Y, Z = makeAggregateMesh(listOfSingleColorFrames)
            
    #        visualizeColorMesh(X, Y, Z, np.roots(occluder))
    #        visualizeColorMesh(X, Y, Z, [])        
                

        if truncation:
            listOfSingleColorFrames = getListOfSingleColorFrames(listOfConvolvedDifferenceFrames)[:]

            listOfTruncatedSingleColorFrames = truncate(listOfSingleColorFrames)        

            print listOfTruncatedSingleColorFrames.shape

            truncatedDifferenceFrames = np.concatenate(imageify(listOfTruncatedSingleColorFrames), 1)

            displayConcatenatedArray(truncatedDifferenceFrames, magnification=100, \
                differenceImage=True, rowsPerFrame=1, stretchFactor=25)    

            rootsFound = findRoots(listOfTruncatedSingleColorFrames, np.roots(occluder), \
                numRootsExpected=len(occluder)-1, showy=False)

        if singlePoly:
        
            listOfSingleColorFrames = getListOfSingleColorFrames(listOfConvolvedDifferenceFrames)

            randomSingleColorFrameIndex = random.randint(0, len(listOfSingleColorFrames)-1)
        
            seq = listOfSingleColorFrames[randomSingleColorFrameIndex]
        #    displayRoots(occluderPolynomial, [], 0, 1)
     
            THETA, R, Z = makeRadialMesh(seq, densityFunc)
            
            
            
            makeRootMagnitudeHistogram(seq, densitySigma)
            
            displayRoots(Polynomial(seq[::-1]),\
                [(i, 1) for i in occluderPolynomial.roots()], 0.5, 1)

            X, Y, Z = makeFuzzyPolynomialMesh(seq)
        
            visualizeColorMesh(X, Y, Z, np.roots(seq))
            visualizeColorMesh(X, Y, Z, [])
        
            
        if movieRecovery:
            recoveredPoly = occluder
            lenPoly = len(recoveredPoly)

            toeplitzMatrix = toeplitz(recoveredPoly[:int((lenPoly+1)/2)][::-1], recoveredPoly[int((lenPoly-1)/2):])
            inverseToeplitzMatrix = np.linalg.inv(toeplitzMatrix)

            recoveredMovie = doFuncToEachChannel(lambda x: np.transpose(np.dot(inverseToeplitzMatrix, np.transpose(x))), concatenatedDifferenceFrames)


    #    visualizePolynomialValues(listOfSingleColorFrames[randomSingleColorFrameIndex])

    if DENSITY_TESTS:    
        sigma = 0.04
        densityFunc = lambda x: logLaplace(x, 0, sigma)
        
        numPointsParam = 10
        
    #    evenPoints = makeEvenPointsAccordingToDensity(densityFunc, numPointsParam, 0, \
    #        100, 1, 30)
            
        evenPoints = makeEvenPointsAccordingToDensity(densityFunc, 10*numPointsParam, 0, \
            100, 1, 300)

    if GIANT_GRADIENT_DESCENT:
        listOfFlatFrames = batchList(pickle.load(open("flat_frames_coarse.p", "r")), 2)
    #    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)

        listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
            for i in range(len(listOfFlatFrames) - 1)][100:150]

        listOfBWFlatDifferenceFrames = [bwifyFlatFrame(flatFrame) for flatFrame in listOfFlatDifferenceFrames]

    #    print np.concatenate(imageify(np.array(listOfBWFlatDifferenceFrames)), 1).shape

    #    print np.array(listOfBWFlatDifferenceFrames)

        displayConcatenatedArray(np.concatenate(imageify(np.array(listOfBWFlatDifferenceFrames))/255, 1), magnification=100,
            differenceImage=True, rowsPerFrame=1)

        occluder = np.array([1, 0, 0, 1, 1])
    #    occluder = np.array([1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1])

        numFrames = len(listOfBWFlatDifferenceFrames)
        unconvolvedFrameLength = len(listOfBWFlatDifferenceFrames[0])
    #    occluder = generateZeroOneSeq(2*unconvolvedFrameLength-1)

        occluderLength = len(occluder)

        viewFlatFrame((imageify(occluder)))

        def getConvolvedFrame(occluderSeq, movieFrame):
    #        print occluderSeq, movieFrame

    #        viewFlatFrame(imageify(np.convolve(occluderSeq, movieFrame, mode="full")))

            return np.convolve(occluderSeq, movieFrame, mode="full")

        listOfConvolvedDifferenceFrames = [addNoise(getConvolvedFrame(occluder, frame)) \
            for frame in listOfBWFlatDifferenceFrames]

    #    convolvedDifferenceFrames = np.concatenate(listOfConvolvedDifferenceFrames, 1)

    #    print "convo", np.array(listOfConvolvedDifferenceFrames)

        print imageify(np.array(listOfConvolvedDifferenceFrames)).shape

        displayConcatenatedArray(np.concatenate(imageify(np.array(listOfConvolvedDifferenceFrames))/255,1),
            magnification=100, differenceImage=True, rowsPerFrame=1)


        frameLength = len(listOfConvolvedDifferenceFrames[0])

    #    assert occluderLength == 2*frameLength-1

        vecLength = occluderLength + numFrames*frameLength
        initVec = [np.random.normal() for _ in range(vecLength)]

        lambdas = [
            100, # occluder [0,1] enforcement (L1)
            3, # occluder low spatial variability (L1)
            3, # movie low temporal variability (L1)
            10, # movie sparsity enforcement (L1)
            3, # movie low spatial variability (L1)
            1/NOISE_SIGMA # observation-matching enforcement (L2)
        ]

        evaluateSequence = evaluateSequenceMaker(occluderLength, listOfConvolvedDifferenceFrames, \
            getConvolvedFrame, lambdas)

        xopt, nfeval, rc = \
            fmin_tnc(evaluateSequence, initVec, approx_grad=1, maxfun=10000)

        recoveredOccluder = xopt[:occluderLength]

        viewFlatFrame(imageify(recoveredOccluder))




        print "xopt", xopt
        print "nfeval", nfeval
        print "rc", rc

    if ROADSIDE_DECONVOLUTION:
        listOfFlatFrames = batchList(pickle.load(open("flat_frames_fine.p", "r")), 2)

        listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
            for i in range(len(listOfFlatFrames) - 1)][:]

    #    viewFlatFrame(imageify(bwifyFlatFrame(listOfFlatFrames[30]))/255)

    #    viewFlatFrame(listOfFlatFrames[30])

        d = 0.3

        randomIndex = random.randint(0, len(listOfFlatDifferenceFrames)-1)
        print randomIndex

        randomFlatFrame = bwifyFlatFrame(listOfFlatDifferenceFrames[randomIndex])

        frameLength = len(randomFlatFrame)

        occluder = generateZeroOneSeq(frameLength)

        sceneFunc = sequenceToFunction(randomFlatFrame)
        occFunc = sequenceToFunction(occluder)

        viewFlatFrame(imageify(randomFlatFrame))
        viewFlatFrame(imageify(occluder))

        obs = roadsideConvolve(sceneFunc, occFunc, d)

        print obs

        viewFrame(imageify(obs))

    if POLYNOMIAL_EXPERIMENT:

    #    listOfFlatFrames = pickle.load(open("real_flat_frames_2500_4500_downsampled.p", "r"))
        listOfFlatFrames = pickle.load(open("flat_frames_grey_bar.p", "r"))


        print "list loaded"

        print len(listOfFlatFrames)

        print listOfFlatFrames[0].shape

        print listOfFlatFrames[1].shape

        downsampledFrames = [batchArrayAlongAxis(frame, 0, 1) for frame in listOfFlatFrames]

        listOfFlatDifferenceFrames = [downsampledFrames[i+1] - downsampledFrames[i] \
            for i in range(len(downsampledFrames) - 1)][:200]

        print len(listOfFlatDifferenceFrames)


        concatenatedOriginalFrames = np.swapaxes(np.concatenate(imageify(np.repeat(np.array(listOfFlatFrames), 1, axis=1))/255, 1), 0, 1)
        concatenatedDifferenceFrames = np.swapaxes(np.concatenate(imageify(np.repeat(np.array(listOfFlatDifferenceFrames), 1, axis=1))/255, 1), 0,1)


        print np.array([listOfFlatDifferenceFrames[0]]*100).shape

    #    viewFrame(np.array([listOfFlatDifferenceFrames[0]]*100), magnification=100, differenceImage=True)

    #    print concatenatedOriginalFrames.shape

    #    displayConcatenatedArray(concatenatedOriginalFrames, magnification=100, \
    #        differenceImage=True, rowsPerFrame=10)
        print concatenatedDifferenceFrames.shape
     
        viewFrame(concatenatedOriginalFrames, magnification=1, differenceImage=True)
        viewFrame(concatenatedDifferenceFrames, magnification=100, differenceImage=True)

        listOfSingleColorFrames = getListOfSingleColorFrames(listOfFlatDifferenceFrames)[:]

    #    eyeballedRoots = [(-0.39, -1.08),
    #                (-0.15, -0.97),
    #                (0.55, -0.77),
     #               (0.99, -0.49),
     #               (-0.69, -0.66),
     #               (-0.82, -0.42),
     #               (-0.99, -0.25),
      #              (0.12, -0.87),
       #             (0.66, -0.31),
        #            (-1.02, 0)]               

        eyeballedRoots = [(0.86, -0.21),
                            (0.85, -0.49),
                            (0.73, -0.81),
                            (0.49, -0.86),
                            (0.15, -1.04),
                            (-0.14, -1.05),
                            (-0.35, -0.91),
                            (-0.75, -0.73),
                            (-0.85, -0.44),
                            (-0.98, -0.26),
                            (-1.11, 0)]


        eyeballedRootsComplexForm = [i[0] + 1j*i[1] for i in eyeballedRoots] + [i[0] + -1j*i[1] for i in eyeballedRoots[:-1]]

        rootsFound = findRoots(listOfSingleColorFrames, eyeballedRootsComplexForm, \
            numRootsExpected=19, showy=False)

    #    print rootsFound

    #    rootsFound = eyeballedRootsComplexForm

    #    complexFormRootsFound = [root[0] + 1j*root[1] for root in rootsFound]

    #    print "correct roots", np.roots(occluder)
    #    print "recovered roots", complexFormRootsFound
    #    print "correct polynomial", polyfromroots(np.roots(occluder))
    #    print "recovered polynomial", polyfromroots(complexFormRootsFound)

    #        print rootsFound

    #        complexFormRootsFound = [np.exp(1j*root[0])*root[1] for root in rootsFound]

        fitPoly = polyfromroots(rootsFound)/2

        pickle.dump(fitPoly, open("fit_poly.p", "w"))

        print fitPoly
        
        viewFlatFrame(imageify(fitPoly), differenceImage=False, magnification=1, filename="recovered_occluder.png")


    if POLYNOMIAL_EXPERIMENT_AFTERMATH:

        rootsFound = pickle.load(open("found_roots", "r"))

        for rf in rootsFound:
            print rf

        complexFormRootsFound = [root[0][0] + 1j*root[0][1] for root in rootsFound]

        print complexFormRootsFound

        recoveredRoots = cleanFoundRoots(complexFormRootsFound, 19)    

        for root in complexFormRootsFound:
            p.plot(np.real(root), np.imag(root), "ro")
           

        for root in recoveredRoots:
            p.plot(np.real(root), np.imag(root), "bo")


        p.show()

        fitPoly = polyfromroots(recoveredRoots)

        pickle.dump(fitPoly, open("fit_poly.p", "w"))

        print fitPoly

        viewFlatFrame(imageify(fitPoly), differenceImage=False, magnification=0.5, filename="recovered_occluder.png")


    #    displayConcatenatedArray(concatenatedDifferenceFrames, magnification=100, \
    #        differenceImage=True, rowsPerFrame=10, stretchFactor=1)    

    if POLYNOMIAL_EXPERIMENT_FRANKENSTEIN:
        poly1 = pickle.load(open("fit_poly_1.p", "r"))

        viewFlatFrame(imageify(poly1))

        poly2 = pickle.load(open("fit_poly_2.p", "r"))

        viewFlatFrame(imageify(poly2))

        fullOccluder = np.concatenate((poly2, poly1), axis=0)

        viewFlatFrame(imageify(fullOccluder))

    if POLYNOMIAL_EXPERIMENT_RECOVERY:
        listOfFlatFrames = pickle.load(open("real_flat_frames_2500_4500_downsampled.p", "r"))

        print "list loaded"

        print len(listOfFlatFrames)

        print listOfFlatFrames[0].shape

        print listOfFlatFrames[1].shape

        downsampledFrames = [batchArrayAlongAxis(frame, 0, 11) for frame in listOfFlatFrames]

        listOfFlatDifferenceFrames = [downsampledFrames[i+1] - downsampledFrames[i] \
            for i in range(len(downsampledFrames) - 1)][300:]

        print len(listOfFlatDifferenceFrames)

    #    concatenatedOriginalFrames = np.swapaxes(np.concatenate(imageify(np.repeat(np.array(listOfFlatFrames), 10, axis=1))/255, 1), 0, 1)
        concatenatedDifferenceFrames = np.swapaxes(np.concatenate(imageify(np.repeat(np.array(listOfFlatDifferenceFrames), 1, axis=1))/255, 1), 0,1)


        print np.array([listOfFlatDifferenceFrames[0]]*100).shape

    #    viewFrame(np.array([listOfFlatDifferenceFrames[0]]*100), magnification=100, differenceImage=True)

    #    print concatenatedOriginalFrames.shape

        displayConcatenatedArray(concatenatedDifferenceFrames, magnification=100, \
            differenceImage=True, rowsPerFrame=10)
     #   print concatenatedDifferenceFrames.shape
     
    #    viewFrame(concatenatedOriginalFrames, magnification=1, differenceImage=True)
    #    viewFrame(concatenatedDifferenceFrames, magnification=100, differenceImage=True)

        listOfSingleColorFrames = getListOfSingleColorFrames(listOfFlatDifferenceFrames)[:]

        recoveredPoly = np.flip(np.concatenate((np.real(pickle.load(open("fit_poly_1.p", "r"))), np.array([0])), axis=0), 0)
        viewFlatFrame(imageify(recoveredPoly))

        lenPoly = len(recoveredPoly)

        print lenPoly

        toeplitzMatrix = toeplitz(recoveredPoly[:int((lenPoly+1)/2)][::-1], recoveredPoly[int((lenPoly-1)/2):])

        print toeplitzMatrix.shape

        inverseToeplitzMatrix = np.linalg.inv(toeplitzMatrix)

        recoveredMovie = doFuncToEachChannel(lambda x: np.transpose(np.dot(inverseToeplitzMatrix, np.transpose(x))), concatenatedDifferenceFrames)

        viewFrame(np.repeat(recoveredMovie, 10, axis=1), magnification=100, differenceImage=True)

        p.matshow(toeplitzMatrix)
        p.show()

    if SIMPLE_BLIND_DECONV_TEST:

        n = 100

        numSamples = 1000
        listOfFreqs = []

        for _ in range(numSamples):
     
            randomNumbers = generateGaussianSeq(n, 1)
     #       randomNumbers = [random.random()*sqrt(12) for _ in range(n)]
     #       randomNumbers = [2*(random.random()>0.5) for _ in range(n)]
    #        randomNumbers = generateZeroOneSeq(n)
    #        randomNumbers = generateSparseSeq(n)

            fftNum = np.fft.fft(randomNumbers)/sqrt(n)
            listOfFreqs.extend([abs(i) for i in fftNum])

    #        listOfFreqs.append(abs(randomNumbers[0]))
        
        numBins = 200
        maxVal = 3
        xVals = np.linspace(0, maxVal, numBins)

        av = average(listOfFreqs)

        print av
        offset = 2

    #    gammaDist = gammaDistMaker(offset, av/offset)
        absNormal = absNormalMaker(sqrt(2))
    #    absNormal = absNormalMaker(5)

        invAbsNormal = invAbsNormalMaker(sqrt(2), 2)


        print quad(lambda x: x*absNormal(x), 0, np.inf)
        print quad(invAbsNormal, 0, np.inf)

        p.plot(xVals, [invAbsNormal(x) for x in xVals])
        p.plot(xVals, [absNormal(x) for x in xVals])

        p.hist(listOfFreqs, bins=200, range=(0, 3), density=True)
        p.show()

    if RECOVER_FREQ_MAGNITUDES:
        n = 100
        movieLength = 1000

        numBins = 200
        maxVal = 3
        xVals = np.linspace(0, maxVal, numBins)

        gaussianOccluder = generateGaussianSeq(n, 1)
        occluderFreqs = np.fft.fft(gaussianOccluder)/sqrt(n)
        trueMovie = [generateGaussianSeq(n, 1) for _ in range(movieLength)]

        convolvedMovie = [wrapConvolve(gaussianOccluder, frame) for frame in trueMovie]
        observedMovie = [addNoise(frame) for frame in convolvedMovie]

        freqDistribs = [[absNormalMaker(sqrt(2))] for _ in range(n)]

        freqs = []
        for trueFrame in trueMovie:
            freqs.extend(np.abs(np.fft.fft(trueFrame))/sqrt(n))
    #        freqs.extend(np.fft.fft(trueFrame)/sqrt(n))

        p.hist(freqs, bins=numBins, range=(0, maxVal), density=True)
        p.plot(xVals, [absNormalMaker(sqrt(2))(x) for x in xVals])

        p.show()

        movieFreqs = []
        print "trueFreq", np.abs(occluderFreqs[1])
        trueFreq = np.abs(occluderFreqs[1])

        for j, observedFrame in enumerate(observedMovie):
            observedFreqs = np.fft.fft(observedFrame)/sqrt(n)
            
            print "movieFreq", abs(np.fft.fft(trueMovie[j])[1]/sqrt(n))
            movieFreq = abs(np.fft.fft(trueMovie[j])[1]/sqrt(n))

            finalFreqDistribs = [distribProductMaker(listOfDistribs) for listOfDistribs in freqDistribs]
    #        p.plot(xVals, [finalFreqDistribs[1](x) for x in xVals])
    #        p.show()        

            for i, freq in enumerate(observedFreqs):
                if i == 1:
                    print "obsFreq", np.abs(freq), abs(np.fft.fft(trueMovie[j])[1]/sqrt(n))*trueFreq

     #               print fmin(lambda x: -invAbsNormalMaker(sqrt(2), abs(freq))(x), \
     #                   np.abs(trueFreq)) 

    #                p.plot(xVals, [invAbsNormalMaker(sqrt(2)*2/sqrt(pi), abs(freq))(x) for x in xVals])
    #                p.plot(xVals, [invAbsNormalMaker(sqrt(2)/sqrt(3/2), abs(freq))(x) for x in xVals])

    #                p.axvline(x=abs(trueFreq), color="green")
    #                p.show()            

                freqDistribs[i].append(invAbsNormalMaker(sqrt(3), np.abs(freq)))

        finalFreqDistribs = [logDistribProductMaker(listOfDistribs) for listOfDistribs in freqDistribs]
        for i in range(n):

            print i, np.abs(occluderFreqs[i]), fmin(lambda x: -finalFreqDistribs[i](x), \
                np.abs(occluderFreqs[i])) 
            p.plot(xVals, [finalFreqDistribs[i](x) for x in xVals])
            p.axvline(x=abs(occluderFreqs[i]), color="red")        
            ax = p.gca()
            ax.set_ylim(-10000, 10000)


            p.show()


    #        if observedFreqs

    if RECOVER_FREQ_MAGNITUDES_2:
        n = 25
        movieLength = 10000

        numBins = 200
        maxVal = 3
        xVals = np.linspace(0, maxVal, numBins)

    #    predictedMean = pi*sqrt(pi)/2
        predictedMean = sqrt(pi)/2

        gaussianOccluder = generateZeroOneSeqIndep(n)
        occluderFreqs = np.abs(np.fft.fft(gaussianOccluder)/sqrt(n))
        trueMovie = [generateGaussianSeq(n,1) for _ in range(movieLength)]

        convolvedMovie = [wrapConvolve(gaussianOccluder, frame) for frame in trueMovie]
        observedMovie = [addNoise(frame) for frame in convolvedMovie]

        freqDistribs = [[absNormalMaker(sqrt(2))] for _ in range(n)]    

        convolvedMovie = [wrapConvolve(gaussianOccluder, frame) for frame in trueMovie]
        observedMovie = [addNoise(frame) for frame in convolvedMovie]

        freqObs = np.array([np.abs(np.fft.fft(frame)/sqrt(n)) for frame in observedMovie])
        freqObsT = np.transpose(freqObs)

        averageFreqs = [average(freq) for freq in freqObsT]
        estimatedFreqs = np.array([averageFreqs[0]/(predictedMean/1.1)] + \
            [freq/predictedMean for freq in averageFreqs[1:]])

        viewFrame(imageify(np.array(observedMovie[:100])), magnification=1)

     #   estimatedOccluder, solutionFound = retrievePhase(estimatedFreqs)
     #   print "Do we think we found a solution?", solutionFound


     #   viewFlatFrame(imageify(estimatedOccluder), magnification=6)
     #   viewFlatFrame(imageify(gaussianOccluder))
     #   print estimatedOccluder
     #   print gaussianOccluder


    #    print occluderFreqs
     #   print estimatedFreqs
        offset = np.divide(occluderFreqs, estimatedFreqs)

    #    print offset
        p.hist(offset, bins=100, range=(0.85,1.15))
        p.hist(offset, bins=100)
        p.show()



     #   viewFlatFrame(imageify(occluderFreqs), magnification=1)
      #  viewFlatFrame(imageify(np.array(estimatedFreqs)), magnification=1)

    if BLIND_2D_SIM:
        x = 100
        y = 100

        truthFrame = generateSparseFrame(x, y)
        blurryTruthFrame = batchAndDifferentiate(truthFrame, \
            [(25, False), (25, False)])
        viewFrame(imageify(truthFrame))
        viewFrame(imageify(blurryTruthFrame), magnification=1e1)

        occ = generateRandomCorrelatedOccluder(x, y)
        blurryOcc = batchAndDifferentiate(occ, \
            [(25, False), (25, False)])
        viewFrame(imageify(occ))
        viewFrame(imageify(blurryOcc))

        convolvedFrame = convolve2D(truthFrame, occ)
        blurryConvolvedFrame = batchAndDifferentiate(convolvedFrame, \
            [(25, False), (25, False)])

        otherBlurryConvolvedFrame = convolve2D(blurryTruthFrame, blurryOcc)

        print otherBlurryConvolvedFrame.shape, otherBlurryConvolvedFrame

        viewFrame(imageify(convolvedFrame), magnification=3e-2)
        viewFrame(imageify(blurryConvolvedFrame), magnification=3e-2)
        viewFrame(imageify(otherBlurryConvolvedFrame), magnification=1e1)

        mags = getMags(blurryOcc)

        counter = 0
        while True:
            solution, foundSolution, errors = retrievePhase2D(mags, tol=1e-10)
            counter += 1
            print counter

            solution = blurryOcc

    #        viewFrame(imageify(solution))
            resultingFrame = deconvolve2D(blurryConvolvedFrame, solution)
    #        viewFrame(imageify(resultingFrame))
            if sparsity(resultingFrame) > 6:
                break

        viewFrame(imageify(solution))
        viewFrame(imageify(resultingFrame))



    #    viewFrame(imageify(generateSparseFrame(x, y)))
    #    viewFrame(imageify(generateRandomCorrelatedOccluder(x, y)))

    if STUPID_METHOD_2D:
    #    path = "smaller_movie.mov"
    #    vid = imageio.get_reader(path,  'ffmpeg')

    #    numFrames = len(vid)

    #    for i in range(numFrames):
     #       if i % 50 == 0:

    #            im = vid.get_data(i)
    #            frame = np.array(im).astype(float)
    #
    #            viewFrame(frame)
        vid = pickle.load(open("smaller_movie_batched_diff_framesplit.p", "r"))

        frameDims = [90,160]

    #    frameDims = vid[0].shape

     #   print frameDims

    #    frameDims = [10,10]

    #    occDims = [2*i-1 for i in frameDims]
    #    occ = generateRandomVeryCorrelatedOccluder(occDims[0], occDims[1], 13)

        occ = pickle.load(open("corr_occ.p", "r"))
        
    #    occ = generateRandomCorrelatedOccluder(occDims[0], occDims[1])

        viewFrame(imageify(occ))

    #    pickle.dump(occ, open("corr_occ.p", "w"))

    #    for i, frame in enumerate(vid):
    #        if i % 50 == 0:
    #            viewFrame(imageify(occ))
    #            viewFrame(imageify(frame)/20)
     #           viewFrame(imageify(convolve2DToeplitz(occ, frame))/20)

    #    frame1 = random.choice(vid)
    #    frame2 = random.choice(vid)

        frame1 = vid[1200]
        frame2 = vid[200]

        frame1Stretched = stretchArray(frame1, (256, 256))
        frame1Sparsified = sparsify(frame1Stretched, 0.3)

        print frame1Stretched.shape
        viewFrame(imageify(frame1Sparsified), adaptiveScaling=True, differenceImage=True)

        pickle.dump(frame1Sparsified, open("sparse_vickie_movement.p", "w"))

    #    frame1 = makeRandomImpulseFrame(frameDims)
     #   frame2 = makeRandomImpulseFrame(frameDims)

    #    frame1 = makeMultipleImpulseFrame(frameDims, 2)
    #    frame2 = makeMultipleImpulseFrame(frameDims, 2)

    #    frame1 = makeImpulseFrame(frameDims, tuple([int(i2) for i in frameDims]))
    #    frame2 = makeImpulseFrame(frameDims, tuple([int(i/2) for i in frameDims]))

    #    frame1 = makeImpulseFrame(frameDims, (45,80))
     #   frame2 = makeImpulseFrame(frameDims, (80,120))

        viewFrame(imageify(frame1), magnification=1, adaptiveScaling=True, differenceImage=True)
        viewFrame(imageify(frame2), magnification=1, adaptiveScaling=True, differenceImage=True)

        convolvedFrame1 = convolve2DToeplitz(frame1, occ)
        convolvedFrame2 = convolve2DToeplitz(frame2, occ)

        viewFrame(imageify(convolvedFrame1), magnification=1, adaptiveScaling=True, differenceImage=False)
        viewFrame(imageify(convolvedFrame2), magnification=1, adaptiveScaling=True, differenceImage=False)

        matchArray, bestMatchArray, bestMatchIndex = getMatchArray(np.abs(convolvedFrame1), \
            np.abs(convolvedFrame2))

        print bestMatchIndex

        overlapArray1, overlapArray2 = getOverlapArray(np.abs(convolvedFrame1), \
            np.abs(convolvedFrame2), bestMatchIndex)

    #    viewFrame(convolvedFrame[frameDims])

        viewFrame(imageify(matchArray), \
            magnification=1, adaptiveScaling=True, differenceImage=False)

        viewFrame(imageify(bestMatchArray), adaptiveScaling=True)

        viewFrame(imageify(overlapArray1), adaptiveScaling=True)
        viewFrame(imageify(overlapArray2), adaptiveScaling=True)

        viewFrame(imageify(np.multiply(overlapArray1, overlapArray2)), adaptiveScaling=True)

        print "max", np.amax(matchArray)

        x = 100        
        y = 100    

    if SPLIT_FRAMES:
        vid = pickle.load(open("fan_monitor_rect_diff.p", "r"))

        vid = np.swapaxes(np.swapaxes(vid,2,3),1,2)

        listOfSingleColorFrames = []

        for frame in vid:
            for singleColorFrame in frame:
                listOfSingleColorFrames.append(singleColorFrame)

        pickle.dump(listOfSingleColorFrames, open("fan_monitor_rect_diff_framesplit.p", "w"))        

    if FOURIER_BURST_ACCUMULATION:
        listOfSingleColorFrames = pickle.load(open("smaller_movie_batched_diff_framesplit.p", "r"))

        occ = pickle.load(open("corr_occ.p", "r"))

        listOfFramesToAggregate = []

        for _ in range(20):
            frame = random.choice(listOfSingleColorFrames)

            viewFrame(imageify(convolve2DToeplitzFull(frame, occ)), \
                adaptiveScaling=True)

            listOfFramesToAggregate.append(convolve2DToeplitzFull(frame, occ))

        aggregatedFrame = aggregateFrames(listOfFramesToAggregate, 3)

        print aggregatedFrame
        viewFrame(imageify(aggregatedFrame), adaptiveScaling=True)
        pickle.dump(aggregatedFrame, open("agg_frame", "w"))

    if LOOK_AT_AGG_FRAME:

        aggFrame = pickle.load(open("agg_frame", "r"))

        viewFrame(imageify(aggFrame), adaptiveScaling=True, magnification=1e5)

    if STUPID_METHOD_2D_2:

        vid = pickle.load(open("smaller_movie_batched_diff_framesplit.p", "r"))

        frameDims = [90,160]

        occ = pickle.load(open("corr_occ.p", "r"))
        
    #    occ = generateRandomCorrelatedOccluder(occDims[0], occDims[1])

        viewFrame(imageify(occ))

        frame1 = random.choice(vid)
        frame2 = random.choice(vid)

        viewFrame(imageify(frame1), magnification=1, adaptiveScaling=True, differenceImage=True)
        viewFrame(imageify(frame2), magnification=1, adaptiveScaling=True, differenceImage=True)

        convolvedFrame1 = convolve2DToeplitzFull(frame1, occ)
        convolvedFrame2 = convolve2DToeplitzFull(frame2, occ)

        viewFrame(imageify(convolvedFrame1), magnification=1, adaptiveScaling=True, differenceImage=False)
        viewFrame(imageify(convolvedFrame2), magnification=1, adaptiveScaling=True, differenceImage=False)

        matchArray, bestMatchArray, bestMatchIndex, matchQuality = getMatchArray(np.abs(convolvedFrame1), \
            np.abs(convolvedFrame2))

        print bestMatchIndex
        print matchQuality

        overlapArray1, overlapArray2 = getOverlapArray(np.abs(convolvedFrame1), \
            np.abs(convolvedFrame2), bestMatchIndex)

    #    viewFrame(convolvedFrame[frameDims])

        viewFrame(imageify(matchArray), \
            magnification=1, adaptiveScaling=True, differenceImage=False)

        viewFrame(imageify(bestMatchArray), adaptiveScaling=True)

        viewFrame(imageify(overlapArray1), adaptiveScaling=True)
        viewFrame(imageify(overlapArray2), adaptiveScaling=True)

        viewFrame(imageify(np.multiply(overlapArray1, overlapArray2)), adaptiveScaling=True)

        print "max", np.amax(matchArray)

        x = 100        
        y = 100    

    if STUPID_METHOD_2D_3:
        vid = pickle.load(open("steven_batched_diff_framesplit.p", "r"))
        occ = pickle.load(open("corr_occ_2.p", "r"))

        lenVid = len(vid)

    #    occ = generateRandomVeryCorrelatedOccluder(36, 64, 3)

        viewFrame(imageify(occ))
    #    pickle.dump(occ, open("corr_occ_2.p", "w"))

        frame = random.choice(vid)

        frameShape = frame.shape

        viewFrame(imageify(frame), differenceImage=True, adaptiveScaling=True)

        convolvedFrame = convolve2DToeplitzFull(frame, occ)

        convolvedFrameShape = convolvedFrame.shape
        print convolvedFrameShape

        viewFrame(imageify(convolvedFrame), differenceImage=True, adaptiveScaling=True)

        otherConvolvedFrame = np.reshape(np.dot(getForwardModelMatrix2DToeplitzFull(occ), frame.flatten()), convolvedFrameShape)

        viewFrame(imageify(otherConvolvedFrame), differenceImage=True, adaptiveScaling=True)

        pseudoInverse = getPseudoInverseSmooth(getForwardModelMatrix2DToeplitzFull(occ), convolvedFrameShape, 1e6)

        recoveredFrame = np.reshape(np.dot(pseudoInverse, convolvedFrame.flatten()), frameShape)

        viewFrame(imageify(frame), differenceImage=True, adaptiveScaling=True)

    if STUPID_METHOD_2D_4:
        diffVid = pickle.load(open("steven_batched_diff_framesplit.p", "r"))
    #    diffVid = pickle.load(open("rick_morty_batched_diff_framesplit.p", "r"))

        occ = pickle.load(open("corr_occ_2.p", "r"))
    #    viewFrame(imageify(occ))

        vid = pickle.load(open("steven_batched.p", "r"))
    #    vid = pickle.load(open("rick_morty_batched.p", "r"))


        lenVid = len(vid)   

        firstMatch = getFirstMatch(diffVid, occ, vid)


    #        viewFrame(imageify(convolvedFrame1), adaptiveScaling=True, differenceImage=True)
    #        viewFrame(imageify(convolvedFrame2), adaptiveScaling=True, differenceImage=True)

               

    #        matchArray, bestMatchArray, bestMatchIndex, matchQuality = getMatchArray(np.abs(convolvedFrame1), \
    #            np.abs(convolvedFrame2))

    if ANALYZE_EXTRACTED_OCCLUDER:
        diffVid = pickle.load(open("steven_batched_diff_framesplit.p", "r"))    
        binaryOcc = pickle.load(open("extracted_occ_bin.p"))

        diffFrame = random.choice(diffVid)
        frameShape = diffFrame.shape

        occ = pickle.load(open("corr_occ_2.p", "r"))

        print occ.shape

    #    convolvedDiffFrame = np.abs(addNoise(convolve2DToeplitzFull(diffFrame, occ)))

    #    print convolvedDiffFrame.shape

    #    forwardModelMatrix = getForwardModelMatrix2DToeplitzFull(occ)

    #    inversionMatrix = getPseudoInverse(forwardModelMatrix, 1e1)

    #    recoveredFrame = vectorizedDot(inversionMatrix, convolvedDiffFrame, diffFrame.shape)

    #    viewFrame(imageify(diffFrame), differenceImage=True, adaptiveScaling=True)
    #    viewFrame(imageify(recoveredFrame), differenceImage=True, adaptiveScaling=True)

     #   print "sumabs", np.sum(np.abs(recoveredFrame))

        viewFrame(imageify(binaryOcc))


    #    impulseFrame = makeImpulseFrame(frameShape, tuple([int(i/2) for i in frameShape]))
        impulseFrame = makeMultipleImpulseFrame(frameShape, 10)

        viewFrame(imageify(impulseFrame))

        convolvedFrame = convolve2DToeplitzFull(impulseFrame, occ)

        viewFrame(imageify(convolvedFrame), adaptiveScaling=True)

        pseudoinverse = False
        wienerFilter = True

        if pseudoinverse:
            forwardModelMatrix = getForwardModelMatrix2DToeplitzFull(binaryOcc)

            inversionMatrix = getPseudoInverse(forwardModelMatrix, 1e1)

            recoveredFrame = vectorizedDot(inversionMatrix, convolvedFrame, frameShape)
        if wienerFilter:
            recoveredFrame, _ = restoration.unsupervised_wiener(convolvedFrame, binaryOcc)

        viewFrame(imageify(recoveredFrame), adaptiveScaling=True)

    #    viewFrame(imageify(diffFrame), differenceImage=True, adaptiveScaling=True)
    #    viewFrame(imageify(recoveredFrame), differenceImage=True, adaptiveScaling=True)
        print "sumabs", np.sum(np.abs(recoveredFrame))

        for i, candidateOcc in enumerate(convertOccToZeroOne(recoveredOcc)):
            print i

            forwardModelMatrix = getForwardModelMatrix2DToeplitzFull(candidateOcc)

            inversionMatrix = getPseudoInverse(forwardModelMatrix, 1e1)

            recoveredFrame = vectorizedDot(inversionMatrix, convolvedDiffFrame, diffFrame.shape)

            viewFrame(imageify(candidateOcc))

    #        viewFrame(imageify(diffFrame), differenceImage=True, adaptiveScaling=True)
    #        viewFrame(imageify(recoveredFrame), differenceImage=True, adaptiveScaling=True)

            print "sumabs", np.sum(np.abs(recoveredFrame))

    if CREATE_RECONSTRUCTION_MOVIE:
        vid = pickle.load(open("steven_batched.p", "r"))

        occ = pickle.load(open("corr_occ_2.p", "r"))

        oldStuff = False

        diffVid = pickle.load(open("steven_batched_diff_framesplit.p", "r"))    
        recoveredOcc = pickle.load(open("extracted_occ.p"))
        binaryOcc = pickle.load(open("extracted_occ_bin.p"))
    #    binaryOcc = convertOccToZeroOne(recoveredOcc)[6]

    #    viewFrame(imageify(recoveredOcc), adaptiveScaling=True)
    #    viewFrame(imageify(binaryOcc))

        if oldStuff:

            forwardModelMatrixApproximate = getForwardModelMatrix2DToeplitzFull(binaryOcc)
            inversionMatrixApproximate = getPseudoInverse(forwardModelMatrixApproximate, 1e1)

            forwardModelMatrixClean = getForwardModelMatrix2DToeplitzFull(occ)
            inversionMatrixClean = getPseudoInverse(forwardModelMatrixClean, 1e1)    

        for i, frame in enumerate(vid):
            print i

    #        print doFuncToEachChannel(lambda x: convolve2DToeplitzFull(x, occ), frame)

            obsFrame = addNoise(doFuncToEachChannel(lambda x: convolve2DToeplitzFull(x, occ), frame))

            p.clf()

            p.subplot(221)
            p.axis("off")
            viewFrame(frame, filename="pass", relax=True)
            p.subplot(222)
            p.axis("off")        
            viewFrame(imageify(occ), filename="pass", relax=True)
            p.subplot(224)
            p.axis("off")        
            viewFrame(obsFrame, filename="pass", relax=True, adaptiveScaling=True)

            p.savefig("blind_deconv_movie_sim/frame_" + padIntegerWithZeros(i, 4) + ".png")

            if oldStuff:

                recoveredFrameClean = doFuncToEachChannel(lambda x: vectorizedDot(inversionMatrixClean, x, frame.shape[:-1]), \
                    obsFrame)    

        #        recoveredFrameApproximate = doFuncToEachChannel(lambda x: vectorizedDot(inversionMatrixApproximate, x, \
         #           frame.shape[:-1]), obsFrame)    
        #        recoveredFrameApproximate = doFuncToEachChannel(lambda x: restoration.unsupervised_wiener(x, binaryOcc)[0], \
         #           obsFrame)
        #        recoveredFrameApproximate = doFuncToEachChannel(lambda x: restoration.richardson_lucy(x, binaryOcc, clip=True), \
        #            obsFrame)
                recoveredFrameApproximate = doFuncToEachChannel(lambda x: restoration.wiener(x, binaryOcc, 10), \
                    obsFrame)


                medfiltedRecoveredFrameApproximate = doFuncToEachChannel(medfilt, recoveredFrameApproximate)

                p.clf()

                p.subplot(221)
                viewFrame(frame, filename="pass", relax=True)
                p.subplot(222)
                viewFrame(obsFrame, filename="pass", relax=True, adaptiveScaling=True)
                p.subplot(223)
                viewFrame(recoveredFrameClean, filename="pass", relax=True)
                p.subplot(224)
                viewFrame(medfiltedRecoveredFrameApproximate, filename="pass", relax=True, magnification=1e2)
        #        viewFrame(medfiltedRecoveredFrameApproximate, filename="pass", relax=True, magnification=2e2)


                p.savefig("blind_deconv_movie_wiener/frame_" + padIntegerWithZeros(i, 3) + ".png")

    if VIEW_FRAMES:
        if False:
            vid = pickle.load(open("blind_deconv_cardboard_1_rect.p", "r"))

            frame = vid[5]
            obsFrame = addNoise(doFuncToEachChannel(lambda x: convolve2DToeplitzFull(x, occ), frame))

            viewFrame(vid[5])    
            viewFrame(imageify(occ))
            viewFrame(obsFrame, adaptiveScaling=True)

        vid = pickle.load(open("steven_batched_diff_framesplit.p", "r"))

        occ = pickle.load(open("corr_occ_2.p", "r"))

        frame1 = vid[350]
        obsFrame1 = addNoise(convolve2DToeplitzFull(frame1, occ))

        frame2 = vid[404]
        obsFrame2 = addNoise(convolve2DToeplitzFull(frame2, occ))
     
        matchArray, bestMatchArray, bestMatchIndex, matchQuality = \
            getMatchArray(obsFrame1, obsFrame2)

    #        viewFrame(imageify(matchArray), adaptiveScaling=True)

        overlapArray1, overlapArray2 = getOverlapArray(obsFrame1, \
            obsFrame2, bestMatchIndex)

        frame3 = makeImpulseFrame(frame1.shape, (int(frame1.shape[0]/2), int(frame1.shape[1]/2)))
        obsFrame3 = convolve2DToeplitzFull(frame3, occ)

        viewFrame(imageify(frame3))
        viewFrame(imageify(occ))
        viewFrame(imageify(obsFrame3))

        viewFrame(imageify(frame1), adaptiveScaling=True, differenceImage=True)    
        viewFrame(imageify(frame2), adaptiveScaling=True, differenceImage=True)
        viewFrame(imageify(occ))
        viewFrame(imageify(np.abs(obsFrame1)), adaptiveScaling=True, differenceImage=True)
        viewFrame(imageify(np.abs(obsFrame1)), adaptiveScaling=True, differenceImage=False)
        viewFrame(imageify(np.abs(obsFrame2)), adaptiveScaling=True, differenceImage=False)

        viewFrame(imageify(matchArray), adaptiveScaling=True)
        viewFrame(imageify(overlapArray1), adaptiveScaling=True)
        viewFrame(imageify(overlapArray2), adaptiveScaling=True)

        newVersion = np.multiply(np.sqrt(np.abs(overlapArray1)), np.sqrt(np.abs(overlapArray2)))

        viewFrame(imageify(newVersion), adaptiveScaling=True)

    if DIFF_EXP_VIDEO:
        vid = pickle.load(open("blind_deconv_hourglass_rect.p", "r"))

        diffVid = batchAndDifferentiate(vid[0], [(1, True), (1, False), (1, False), (1, False)])

        pickle.dump(diffVid, open("blind_deconv_hourglass_rect_diff.p", "w"))

    if PROCESS_EXP_VIDEO:
        diffVid = pickle.load(open("blind_deconv_hourglass_rect_diff_framesplit.p", "r"))

        frameShape = diffVid[0].shape

        occ = estimateOccluderFromDifferenceFrames(diffVid, [i*2 for i in frameShape])

        viewFrame(occ, adaptiveScaling=True, differenceImage=False)

    if CONV_SIM_VIDEO:
        diffVidUnconvolved = pickle.load(open("steven_batched_diff_framesplit.p", "r"))

        occ = pickle.load(open("corr_occ_2.p", "r"))

        convVid = []

        for i, frame in enumerate(diffVidUnconvolved):
            print i, "/", len(diffVidUnconvolved)

            convVid.append(convolve2DToeplitzFull(frame, occ)) 

        pickle.dump(np.array(convVid), open("steven_batched_diff_conv_framesplit.p", "w"))

    if PROCESS_SIM_VIDEO:
    #    diffVid = pickle.load(open("steven_batched_diff_conv_framesplit.p", "r"))
    #    diffVid = pickle.load(open("blind_deconv_hourglass_rect_diff_framesplit.p", "r"))
    #    diffVid = pickle.load(open("fan_rect_meansub_framesplit.p", "r"))
        diffVid = pickle.load(open("plant_rect_meansub_pos_framesplit.p", "r"))
    #    diffVid = pickle.load(open("fan_monitor_rect_diff_framesplit.p", ))

        frameShape = diffVid[0].shape

        occ = estimateOccluderFromDifferenceFramesCanvasPreserving(diffVid)

        viewFrame(imageify(occ), adaptiveScaling=True, differenceImage=False)


    if VIEW_OCC:
        occ = pickle.load(open("extracted_occ_exp.p", "r"))    

        viewFrame(occ, adaptiveScaling=True)

    if OVERLAP_PAD_TEST:

        frame1Dims = (10, 20)
        frame2Dims = (20, 10)

        frame1 = makeImpulseFrame(frame1Dims, (4,8))
        frame2 = makeImpulseFrame(frame2Dims, (8,4))

        occ = pickle.load(open("corr_occ_2.p", "r"))
        print occ.shape

        viewFrame(imageify(occ))


        convolvedFrame1 = convolve2DToeplitz(frame1, occ)
        convolvedFrame2 = convolve2DToeplitz(frame2, occ)

        viewFrame(imageify(convolvedFrame1))
        viewFrame(imageify(convolvedFrame2))

        matchArray, bestIndex = getMatchArrayUnequalSize(np.abs(convolvedFrame1), \
            np.abs(convolvedFrame2))

        overlapArray1, overlapArray2 = getOverlapArrayPadded(convolvedFrame1, convolvedFrame2, bestIndex)

        viewFrame(imageify(overlapArray1))
        viewFrame(imageify(overlapArray2))    

    if OVERLAP_PAD_TEST_2:

        frame1Dims = (18, 32)
        frame2Dims = (18, 32)

        frame1 = makeImpulseFrame(frame1Dims, (9, 16))
        frame2 = makeImpulseFrame(frame2Dims, (8, 12))

        occ = pickle.load(open("corr_occ_2.p", "r"))
        print occ.shape

        viewFrame(imageify(occ))

        viewFrame(imageify(frame1))
        viewFrame(imageify(frame2))

        convolvedFrame1 = convolve2DToeplitz(frame1, occ)
        convolvedFrame2 = convolve2DToeplitz(frame2, occ)

        viewFrame(imageify(convolvedFrame1))
        viewFrame(imageify(convolvedFrame2))

        matchArray, _, bestIndex, _ = getMatchArray(np.abs(convolvedFrame1), \
            np.abs(convolvedFrame2))

        overlapArray = getOverlapArrayFullCanvas(convolvedFrame1, convolvedFrame2, bestIndex)

        viewFrame(imageify(overlapArray))

    if CONVERT_ARRAYS_CHRISTOS:


        binOcc = np.swapaxes(pickle.load(open("binary_occ_exp.p", "r")), 0, 1)
        vid = np.swapaxes(pickle.load(open("blind_deconv_cardboard_1_rect.p", "r"))[0], 1, 2)

        viewFrame(imageify(binOcc))
        viewFrame(vid[10], adaptiveScaling=True)



        dic = {}
        dic["occluder_array"] = binOcc
        dic["observations"] = vid

        scipy.io.savemat("experiment_data.mat", dic)

    if PROCESS_EXP_OCCLUDER:
        rawOcc = pickle.load(open("fan_extracted_occ.p", "r"))
        viewFrame(imageify(rawOcc), adaptiveScaling=True)

        medFiltOcc = medfilt(rawOcc)
        viewFrame(imageify(medFiltOcc), adaptiveScaling=True)

        occZeroOne = convertOccToZeroOne(medFiltOcc)

        print rawOcc.shape


    #    viewFrame(imageify(occZeroOne[12]))    

    #    recoveredOcc = cutArrayDownToShape(occZeroOne[8], (63, 36), reverse=True)
        recoveredOcc = cutArrayDownToShape(occZeroOne[8], occZeroOne[8].shape, reverse=True)

        downsampledArray = resizeArray(recoveredOcc, (63, 35))

        print type(downsampledArray)

        viewFrame(imageify(downsampledArray))

        pickle.dump(downsampledArray, open("binary_occ_exp_hourglass.p", "w"))

        forwardModelMatrix = getForwardModelMatrix2DToeplitzFull(downsampledArray)

        inversionMatrix = getPseudoInverse(forwardModelMatrix, 3e-4)

        pickle.dump(inversionMatrix, open("extracted_inverter_exp_hourglass.p", "w"))

    if PROCESS_EXP_OCCLUDER_2:
        rawOcc = pickle.load(open("fan_extracted_occ.p", "r"))
        viewFrame(imageify(rawOcc), adaptiveScaling=True)

        medFiltOcc = medfilt(rawOcc)
        viewFrame(imageify(medFiltOcc), adaptiveScaling=True)

    #    medFiltOcc = rawOcc

        occZeroOne = convertOccToZeroOne(medFiltOcc)

    #    print rawOcc.shape

        i = 22

    #    viewFrame(imageify(occZeroOne[12]))    

    #    recoveredOcc = cutArrayDownToShape(occZeroOne[i], (39, 70), reverse=False)
        recoveredOcc = cutArrayDownToShape(occZeroOne[i], occZeroOne[i].shape, reverse=True)

    #    downsampledArray = resizeArray(recoveredOcc, (40, 70))
        downsampledArray = recoveredOcc

        print type(downsampledArray)

        viewFrame(imageify(downsampledArray))

    #    pickle.dump(downsampledArray, open("binary_occ_exp_fan_monitor.p", "w"))

    if PROCESS_EXP_OCCLUDER_BIN_ONLY:
    #    viewFrame(imageify(occZeroOne[8]))

        downsampledArray = pickle.load(open("binary_occ_exp_fan_monitor.p", "r"))

        viewFrame(imageify(downsampledArray))

        forwardModelMatrix = getForwardModelMatrix2DToeplitzFullFlexibleShape(downsampledArray, \
            downsampledArray.shape, (40, 70), padVal=1)

        inversionMatrix = getPseudoInverse(forwardModelMatrix, 3e-4)

        pickle.dump(inversionMatrix, open("extracted_inverter_fan_monitor.p", "w"))

    if PROCESS_EXP_OCCLUDER_CARDBOARD:
    #    viewFrame(imageify(occZeroOne[8]))

        downsampledArray = pickle.load(open("binary_occ_exp.p", "r"))

        viewFrame(imageify(downsampledArray))

        forwardModelMatrix = getForwardModelMatrix2DToeplitzFullFlexibleShape(downsampledArray, \
            (125, 69), (20, 10), padVal=0)

        inversionMatrix = getPseudoInverse(forwardModelMatrix, 3e-3)

        pickle.dump(inversionMatrix, open("extracted_inverter_exp_cardboard.p", "w"))

    if CREATE_RECONSTRUCTION_MOVIE_EXP:
        groundTruthVid = pickle.load(open("steven_batched.p", "r"))

    #    vid = pickle.load(open("blind_deconv_hourglass_rect_meansub.p", "r"))
    #    vid = pickle.load(open("blind_deconv_cardboard_1_rect.p", "r"))[0]
    #    vid = pickle.load(open("blind_deconv_cardboard_1_rect_sim.p", "r"))
    #    vid = pickle.load(open("blind_deconv_cardboard_1_rect_alt.p", "r"))[0]
        vid = pickle.load(open("cardboard_rect.p", "r"))
    #    vid = pickle.load(open("hourglass_rect.p", "r"))
    #    vid = pickle.load(open("fan_rect_meansub.p", "r"))

    #    inversionMatrix = pickle.load(open("extracted_inverter_exp_hourglass.p", "r"))
        inversionMatrix = pickle.load(open("extracted_inverter_exp_cardboard.p", "r"))
    #    inversionMatrix = pickle.load(open("extracted_inverter_exp.p", "r"))

    #    binaryOcc = pickle.load(open("binary_occ_exp_hourglass.p", "r"))
        binaryOcc = pickle.load(open("binary_occ_exp.p", "r"))
    #    binaryOcc = pickle.load(open("binary_occ_exp_fan.p", "r"))

    #    viewFrame(imageify(binaryOcc))

        paddedOcc = padArrayToShape(binaryOcc, (125, 69))

        viewFrame(np.swapaxes(imageify(paddedOcc), 0, 1))

    #    targetShape = (63, 35)
        targetShape = (82, 44)

    #    gaussKernel = getGaussianKernel(5, 1)
    #    gaussKernelLarge = getGaussianKernel(10, 3)

        recoveredMovie = []

    #    print vid.shape

        for i, obsFrame in enumerate(vid[5:]):
            print i

    #        prunedObsFrame = doFuncToEachChannel(lambda x: cutArrayDownToShape(x, (105, 69)), \
     #           obsFrame) 
    #        prunedObsFrame = doFuncToEachChannel(lambda x: convolve2DToeplitzFull(gaussKernelLarge, x), \
    #            obsFrame)
            prunedObsFrame = obsFrame
    #        prunedObsFrame = doFuncToEachChannel(lambda x: np.pad(x, [(0, 0), (15, 0)], "constant"), obsFrame)

    #        resizedObsFrame = doFuncToEachChannel(lambda x: resizeArray(x, (125, 69)), \
    #            prunedObsFrame)

            resizedObsFrame = np.flip(doFuncToEachChannel(lambda x: resizeArray(x, (125, 69)), \
                prunedObsFrame), 2)

    #        print resizedObsFrame.shape
     #       print binaryOcc.shape

    #        viewFrame(resizedObsFrame, adaptiveScaling=True)
      #      viewFrame(imageify(binaryOcc))

    #        recoveredFrame = doFuncToEachChannel(lambda x: restoration.unsupervised_wiener(x, binaryOcc)[0], \
     #           resizedObsFrame)

     #       viewFrame(resizedObsFrame, adaptiveScaling=True)
     #       viewFrame(imageify(binaryOcc))



    #        recoveredFrame = doFuncToEachChannel(lambda x: restoration.wiener(x, binaryOcc, 1e-5), \
    #            resizedObsFrame)

    #        recoveredFrame = doFuncToEachChannel(lambda x: restoration.richardson_lucy(x, binaryOcc), \
     #           resizedObsFrame)

            recoveredFrame = doFuncToEachChannel(lambda x: vectorizedDot(inversionMatrix, x, targetShape), \
                resizedObsFrame)    

      #      viewFrame(recoveredFrame, differenceImage=True, adaptiveScaling=True, magnification=10)

    #        recoveredFrame = resizedObsFrame

    #        recoveredMovie.append(recoveredFrame)


    #        blurryRecoveredFrame = doFuncToEachChannel(medfilt, dividedRecoveredFrame)
    #        blurryRecoveredFrame = doFuncToEachChannel(lambda x: convolve2DToeplitzFull(gaussKernel, x), \
    #            recoveredFrame)
        
    #        blurryRecoveredFrame = recoveredFrame

            blurryRecoveredFrame = recoveredFrame[25:72, 0:34]

            swappedRecoveredFrame = swapChannels(blurryRecoveredFrame, 0, 2)

     #       viewFrame(np.flip(np.swapaxes(blurryRecoveredFrame, 0, 1), 0), adaptiveScaling=True,
     #           relax=True, magnification=3, differenceImage=True)

    #        if i % 3 == 0:
    #            viewFrame(np.swapaxes(blurryRecoveredFrame, 0, 1), magnification=5e1)

            groundTruthFrame = groundTruthVid[2*i+3]

            p.clf()

            p.subplot(221)
            p.axis("off")
            viewFrame(groundTruthFrame, filename="pass", relax=True, differenceImage=True, magnification=0.5)

            p.subplot(223)
            p.axis("off")
            viewFrame(np.swapaxes(obsFrame, 0, 1), filename="pass", adaptiveScaling=True,
                relax=True, magnification=1, differenceImage=False)

            p.subplot(222)
            p.axis("off")
            viewFrame(np.flip(np.swapaxes(swappedRecoveredFrame, 0, 1), 0), filename="pass", adaptiveScaling=True,
                relax=True, magnification=0.6, differenceImage=True)

            p.savefig("blind_deconv_movie_exp_cardboard/frame_" + padIntegerWithZeros(i, 4) + ".png")

    if CREATE_RECONSTRUCTION_MOVIE_EXP_2:
    #    groundTruthVid = pickle.load(open("steven_batched.p", "r"))

        

        vid = pickle.load(open("plant_rect_meansub.p", "r"))
    #    vid = pickle.load(open("fan_rect_meansub.p", "r"))
    #    vid = pickle.load(open("fan_monitor_fine_rect_meansub.p", "r"))

        inversionMatrix = pickle.load(open("extracted_inverter_exp_plant.p", "r"))
    #    inversionMatrix = pickle.load(open("extracted_inverter_exp_fan.p", "r"))
    #    inversionMatrix = pickle.load(open("extracted_inverter_fan_monitor.p", "r"))

        binaryOcc = pickle.load(open("binary_occ_exp_plant.p", "r"))
    #    binaryOcc = pickle.load(open("binary_occ_exp_fan.p", "r"))
    #    binaryOcc = pickle.load(open("binary_occ_exp_fan_monitor.p", "r"))

        viewFrame(imageify(binaryOcc))

        targetShape = (30, 50)
    #    targetShape = (30, 60)
    #    targetShape = (40, 70)

        gaussKernel = getGaussianKernel(5, 1)
        gaussKernelLarge = getGaussianKernel(10, 3)

        recoveredMovie = []

        for i, obsFrame in enumerate(vid[:]):
            print i

            recoveredFrame = doFuncToEachChannel(lambda x: vectorizedDot(inversionMatrix, x, targetShape), \
                obsFrame)    

    #        groundTruthFrame = groundTruthVid[2*i+3]

            p.clf()

            p.axis("off")
    #        p.subplot(211)
    #        viewFrame(groundTruthFrame, filename="pass", relax=True, differenceImage=True, magnification=0.5)
    #        p.subplot(212)
            viewFrame(recoveredFrame, filename="pass", adaptiveScaling=True,
                relax=True, magnification=1, differenceImage=True)

            p.savefig("blind_deconv_movie_exp_plant/frame_" + padIntegerWithZeros(i, 4) + ".png")



    if EXTRACT_MATRIX_FROM_IMAGE:
        mat = extractMatrixFromBWImage("corr_occ_2.png", 24)

        print mat.shape
        viewFrame(imageify(mat), adaptiveScaling=True)

        pickle.dump(mat, open("corr_occ_2.p", "w"))

    if WIENER_FILTER_TEST:
        occ = pickle.load(open("corr_occ_2.p", "r"))

        convOcc = convolve2DToeplitzFull(occ, occ)

        viewFrame(imageify(occ))

        viewFrame(imageify(convOcc), adaptiveScaling=True)

        deconvolved, _ = restoration.unsupervised_wiener(convOcc, occ)

        viewFrame(imageify(deconvolved), adaptiveScaling=True)

    if CROP:
        arr = pickle.load(open("macarena_dark_fixed_diff_medfilt.p", "r"))

        pickle.dump(arr[:,19:,:,:], open("macarena_dark_fixed_cropped_diff_medfilt.p", "w"))

    if DOWNSAMPLE_VID:
        arrName = "prafull_ball"

        arr = pickle.load(open(arrName + ".p", "r"))

        newArr = []
        downsampleRate = 5

        for i, frame in enumerate(arr):
            if i % downsampleRate == 0:
                newArr.append(frame)

        pickle.dump(np.array(newArr), open(arrName + "_ds.p", "w"))

    if MEAN_SUBTRACTION:

        name = "prafull_ball_ds"

        vid = pickle.load(open(name + ".p", "r"))
#        vid = np.swapaxes(pickle.load(open(name + ".p", "r")), 1, 2) # you probably don't want this

        print vid.shape

        averageFrame = sum(vid, 0)/np.shape(vid)[0]

        print averageFrame.shape

    #    viewFrame(averageFrame, adaptiveScaling=True)

    #    for i, frame in enumerate(vid):
    #        if i % 10 == 0:
    #            viewFrame(frame, adaptiveScaling=True)
     #           viewFrame(frame - averageFrame, adaptiveScaling=True, differenceImage=True)

        meanSubVid = []

        for frame in vid:
            meanSubVid.append(frame - averageFrame)

        pickle.dump(np.array(meanSubVid), open(name + "_meansub.p", "w"))

    if UNIFORM_MEAN_SUBTRACTION:

        vid = pickle.load(open("fan_rect.p", "r"))

        frameShape = vid[0].shape

        print vid.shape

        averageFramePixel = np.sum(np.sum(np.sum(vid, 0), 0), 0)

        print averageFramePixel.shape

        summedFrame = np.array([[averageFramePixel]*frameShape[1]]*frameShape[0])/(len(vid)*frameShape[0]*frameShape[1])    

     #   print summedFrame

    #    averageFrame = sum(vid, 0)/np.shape(vid)[0]

    #    print averageFrame.shape

    #    viewFrame(averageFrame, adaptiveScaling=True)

    #    for i, frame in enumerate(vid):
    #        if i % 10 == 0:
    #            viewFrame(frame, adaptiveScaling=True)
     #           viewFrame(frame - averageFrame, adaptiveScaling=True, differenceImage=True)

        meanSubVid = []

        for frame in vid:
            meanSubVid.append(frame - summedFrame)

        pickle.dump(meanSubVid, open("fan_rect_meansubuniform.p", "w"))    

    if MEAN_SUBTRACTION_POSITIVE:
        vid = np.array(pickle.load(open("plant_rect_meansub.p", "r")))

        mostNegValue = np.min(vid)

        meanSubPosVid = vid - mostNegValue*np.ones(vid.shape)

        pickle.dump(meanSubPosVid, open("plant_rect_meansub_pos.p", "w"))    

    if SIM_COMPARISON:
        groundTruthVid = pickle.load(open("steven_batched.p", "r"))

        occ = pickle.load(open("binary_occ_exp.p", "r"))

    #    viewFrame(np.swapaxes(imageify(occ), 0, 1))
    #    viewFrame(imageify(occ))

        simVid = []

        for frame in groundTruthVid:
            convolvedFrame = doFuncToEachChannel(lambda x: convolve2DToeplitzFull(np.swapaxes(x, 0, 1), occ), 
                np.flip(frame, 0))
    #        convolvedFrame = doFuncToEachChannel(lambda x: convolve2DToeplitzFull(x, occ), 
    #            frame)

            simVid.append(convolvedFrame)
    #        viewFrame(convolvedFrame, adaptiveScaling=True)
    #        viewFrame(np.swapaxes(convolvedFrame, 0, 1), adaptiveScaling=True)


        pickle.dump(np.array(simVid), open("blind_deconv_cardboard_1_rect_sim.p", "w"))

    if CAP_ARR_VALS:
        arr = pickle.load(open("orange_rect_meansub.p", "r"))

        capVal = 10

        for frame in arr:
            i = random.randint(0, arr.shape[0]-1)
            j = random.randint(0, arr.shape[1]-1)

            clampedFrame = np.max(np.min(frame, capVal*np.ones(frame.shape)), -1*capVal*np.ones(frame.shape))

        pickle.dump(clampedFrame, open("orange_rect_meansub_clamped.p", "w"))

    if TIME_DIFF:
        arr = pickle.load(open("circle_square_nowrap_vid.p", "r"))

        diffArr = batchAndDifferentiate(arr, [(1, True), (1, False), (1, False), (1, False)])

        pickle.dump(diffArr, open("circle_square_nowrap_vid_diff.p", "w"))

    if MED_FILT:
        arr = pickle.load(open("bld66_rect_diff.p", "r"))

        medfiltedArr = medfilt(arr)

        pickle.dump(medfiltedArr, open("bld66_rect_diff_medfilt.p", "w"))

    if AVERAGE_DIVIDE:
        arrName = "prafull_ball_ds_meansub"

        arr = pickle.load(open(arrName + ".p", "r"))

        frameDims = arr[0].shape[:-1]
        frameSum = np.zeros(frameDims)

        minFactor = 0.1

        numPixels = frameDims[0]*frameDims[1]

        for frame in arr:
            frameSum += np.abs(np.sum(frame, 2)/3)

        returnArr = []

        frameSumAverage = np.sum(frameSum)/numPixels

#        print frameSum.shape

        minIntensityArray = minFactor*frameSumAverage*np.ones(frameDims)

#        print minIntensityArray.shape

        frameSumMinned = np.maximum(frameSum, minFactor*frameSumAverage*np.ones(frameDims))

        intensityBaselineFrame = imageify(frameSumMinned)/255

        for i, frame in enumerate(arr):

            if i == 250:
#                viewFrame(frame, differenceImage=True, adaptiveScaling=True)
                viewFrame(np.divide(frame, intensityBaselineFrame), \
                    magnification=30000, differenceImage=True, adaptiveScaling=False)

            returnArr.append(np.divide(frame, intensityBaselineFrame))

        pickle.dump(returnArr, open(arrName + "_avgdiv.p", "w"))

    if AVERAGE_DIVIDE_1D:
        arr = pickle.load(open("circle_square_nowrap_vid_obs_jumbled_recovery_grouped_meansub_abs_coloravg.p", "r"))

        frameDims = arr[0].shape[:-1]
        frameSum = np.zeros(frameDims)

        minFactor = 0.01

        numPixels = frameDims[0]

        for frame in arr:
            frameSum += np.abs(np.sum(frame, 1)/3)

        returnArr = []

        frameSumAverage = np.sum(frameSum)/numPixels

#        print frameSum.shape

        minIntensityArray = minFactor*frameSumAverage*np.ones(frameDims)

#        print minIntensityArray.shape

        frameSumMinned = np.maximum(frameSum, minFactor*frameSumAverage*np.ones(frameDims))

        intensityBaselineFrame = imageify(frameSumMinned)/255

        for i, frame in enumerate(arr):

            if i == 250:
#                viewFrame(frame, differenceImage=True, adaptiveScaling=True)
                viewFrame(np.divide(frame, intensityBaselineFrame), \
                    magnification=30000, differenceImage=True, adaptiveScaling=False)

            returnArr.append(np.divide(frame, intensityBaselineFrame))

        pickle.dump(returnArr, open("circle_square_nowrap_vid_obs_jumbled_recovery_grouped_meansub_abs_coloravg_avgdiv.p", "w"))

    if GET_ABS:
        arrName = "circle_square_nowrap_vid_obs_jumbled_recovery_grouped_meansub"

        arr = pickle.load(open(arrName + ".p", "r"))

        returnArr = []

        for frame in arr:
            returnArr.append(np.abs(frame))

        pickle.dump(returnArr, open(arrName + "_abs.p", "w"))

    if COLOR_AVG:
        arrName = "circle_square_nowrap_vid_obs_jumbled_recovery_grouped_meansub_abs"

        arr = pickle.load(open(arrName + ".p", "r"))

        returnArr = []

        for frame in arr:
#            print frame.shape
            newFrame = imageify(np.sum(frame, 2)/(3*255))
#            newFrame = imageify(np.sum(frame, 1)/(3*255))

            returnArr.append(newFrame)

        pickle.dump(returnArr, open(arrName + "_coloravg.p", "w"))

    if COLOR_FLATTEN:
        arrName = "circle_square_nowrap_vid_obs_jumbled_recovery_grouped_meansub_abs_coloravg_avgdiv"

        arr = pickle.load(open(arrName + ".p", "r"))

        returnArr = []

        for frame in arr:
#            print frame.shape
            newFrame = np.sum(frame, 2)/3
#            newFrame = np.sum(frame, 1)/3

            returnArr.append(newFrame)

        pickle.dump(returnArr, open(arrName + "_colorflat.p", "w"))

    if MAKE_VIDEO:
#        arrName = "circle_carlsen_nowrap_vid_meansub_abs_coloravg"
#        arrName = "circle_carlsen_nowrap_vid"
#        arrName = "circle_square_nowrap_vid_meansub_abs"
#        arrName = "circle_carlsen_nowrap_vid_meansub_abs_coloravg_avgdiv"
        arrName = "recovered_vid"
#        arrName = "prafull_ball_meansub_avgdiv"
#        arrName = "circle_square_nowrap_vid_obs"

        arr = np.array(pickle.load(open(arrName + ".p", "r")))

        print arr.shape

#        print arr

#        convertArrayToVideo(np.array(arr), 30000, arrName, 15, adaptiveScaling=False, differenceImage=True)
        convertArrayToVideo(np.array(arr), 0.5, arrName, 15, adaptiveScaling=True, differenceImage=True)

    if DOWNSIZE_ARR:
        arr = pickle.load(open("36225_bright_fixed_rect_diff_medfilt.p", "r"))

        newArr = []

        for i, frame in enumerate(arr):
            print i
            newArr.append(resizeArray(arr, (80, 80)))

        pickle.dump(np.array(newArr), open("36225_bright_fixed_rect_diff_medfilt_downsized.p", "w"))

    if MACARENA_CORRECT_DECONV_DUMPER:
        scene = pickle.load(open("macarena_scene_gt.p", "r"))
        vid = pickle.load(open("macarena_dark_fixed_cropped_meansub_medfilt.p", "r"))

        resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), scene)


    #    print scene.shape
    #    viewFrame(resizedScene)

        print vid[0].shape

        forwardModelMatrices = doFuncToEachChannelSeparated(lambda x: \
            getForwardModelMatrix2DToeplitzFullFlexibleShape(x, vid[0].shape, (10, 10)), resizedScene)

        inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1.5e-8) for forwardModelMatrix in forwardModelMatrices]

    #    print inversionMatrices[0].shape

        pickle.dump(inversionMatrices, open("macarena_inverter_gt.p", "w")) 

    if MACARENA_EXP_DECONV_DUMPER:
        scene = pickle.load(open("inverter_exp_36225.p", "r"))
        vid = pickle.load(open("36225_bright_fixed_rect_diff_medfilt.p", "r"))

        viewFrame(scene, adaptiveScaling=True, differenceImage=True)


    #    resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), scene)
        resizedScene = cutArrayDownToShapeWithAlternateArray(scene, np.sum(np.abs(scene), axis=2), (40, 40))

        resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), resizedScene)

        viewFrame(resizedScene, adaptiveScaling=True, differenceImage=True)

    #    print scene.shape
    #    viewFrame(resizedScene)

    #    print vid[0].shape

        forwardModelMatrices = doFuncToEachChannelSeparated(lambda x: \
            getForwardModelMatrix2DToeplitzFullFlexibleShape(x, vid[0].shape, (10, 10)), resizedScene)

    #    inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1.5e-8) for forwardModelMatrix in forwardModelMatrices]
        inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1e4) for forwardModelMatrix in forwardModelMatrices]

    #    print inversionMatrices[0].shape

        pickle.dump(inversionMatrices, open("macarena_inverter_exp.p", "w")) 

    if EXP_63225_DECONV_DUMPER:
        scene = pickle.load(open("scene_exp_63225.p", "r"))
        vid = pickle.load(open("36225_bright_fixed_rect_diff_medfilt.p", "r"))

        viewFrame(scene, adaptiveScaling=True, differenceImage=True)


    #    resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), scene)
        resizedScene = cutArrayDownToShapeWithAlternateArray(scene, np.sum(np.abs(scene), axis=2), (25, 25))

        resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), resizedScene)

        viewFrame(resizedScene, adaptiveScaling=True, differenceImage=True)

    #    print scene.shape
    #    viewFrame(resizedScene)

    #    print vid[0].shape

        forwardModelMatrices = doFuncToEachChannelSeparated(lambda x: \
            getForwardModelMatrix2DToeplitzFullFlexibleShape(x, (75,75), (10, 10)), resizedScene)

    #    inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1.5e-8) for forwardModelMatrix in forwardModelMatrices]
        inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1e5) for forwardModelMatrix in forwardModelMatrices]

    #    print inversionMatrices[0].shape

        pickle.dump(inversionMatrices, open("inverter_exp_36225.p", "w"))     

    if ORANGE_DECONV_DUMPER:
        scene = pickle.load(open("orange_scene_exp.p", "r"))
        vid = pickle.load(open("orange_rect_meansub_medfilt.p", "r"))

        viewFrame(scene, adaptiveScaling=True, differenceImage=True)


    #    resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), scene)
        resizedScene = cutArrayDownToShapeWithAlternateArray(scene, np.sum(np.abs(scene), axis=2), (60, 60))

    #    resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), resizedScene)

        viewFrame(resizedScene, adaptiveScaling=True, differenceImage=True)

    #    print scene.shape
    #    viewFrame(resizedScene)

    #    print vid[0].shape

        forwardModelMatrices = doFuncToEachChannelSeparated(lambda x: \
            getForwardModelMatrix2DToeplitzFullFlexibleShape(x, (75,75), (50, 50)), resizedScene)

        for i, forwardModelMatrix in enumerate(forwardModelMatrices):
            inversionMatrix = getPseudoInverse(forwardModelMatrix, 1e5) 

            pickle.dump(inversionMatrix, open("orange_inverter_exp_" + str(i) + ".p", "w"))

    #    inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1.5e-8) for forwardModelMatrix in forwardModelMatrices]
    #    inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1e5) for forwardModelMatrix in forwardModelMatrices]

    #    print inversionMatrices[0].shape

    #    pickle.dump(inversionMatrices, open("orange_inverter_exp.p", "w"))     

    if BLD66_DECONV_DUMPER:
        scene = pickle.load(open("bld66_scene_exp.p", "r"))
        vid = pickle.load(open("bld66_rect_meansub.p", "r"))

        viewFrame(scene, adaptiveScaling=True, differenceImage=True)


    #    resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), scene)
    #    resizedScene = cutArrayDownToShapeWithAlternateArray(scene, np.sum(np.abs(scene), axis=2), (20, 40))

        resizedScene = scene[12:33,10:50]

    #    resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), resizedScene)

        viewFrame(resizedScene, adaptiveScaling=True, differenceImage=True)

    #    print scene.shape
    #    viewFrame(resizedScene)

    #    print vid[0].shape

        forwardModelMatrices = doFuncToEachChannelSeparated(lambda x: \
            getForwardModelMatrix2DToeplitzFullFlexibleShape(x, (40,70), (20, 40)), resizedScene)

    #    for i, forwardModelMatrix in enumerate(forwardModelMatrices):
    #        inversionMatrix = getPseudoInverse(forwardModelMatrix, 1e5) 

    #        pickle.dump(inversionMatrix, open("orange_inverter_exp_" + str(i) + ".p", "w"))

    #    inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1.5e-8) for forwardModelMatrix in forwardModelMatrices]
        inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1e5) for forwardModelMatrix in forwardModelMatrices]

    #    print inversionMatrices[0].shape

        pickle.dump(inversionMatrices, open("bld66_inverter_exp.p", "w"))     

    if MACARENA_TEST:
        scene = pickle.load(open("macarena_scene_gt.p", "r"))
        vid = pickle.load(open("macarena_dark_fixed_meansub.p", "r"))

        resizedScene = doFuncToEachChannel(lambda x: resizeArray(x, (25, 25)), scene)

        obsShape = (86, 48)


    #    print scene.shape
    #    viewFrame(resizedScene)

    #    print vid[0].shape

        forwardModelMatrices = doFuncToEachChannelSeparated(lambda x: \
            getForwardModelMatrix2DToeplitzFullFlexibleShape(x, obsShape), resizedScene)

        impulseFrame = makeMultipleImpulseFrame((61, 23), 10)

        viewFrame(imageify(impulseFrame))

        viewFrame(imageify(vectorizedDot(forwardModelMatrices[0], impulseFrame, obsShape)), adaptiveScaling=True)



        convolvedFrame = doSeparateFuncToEachChannel([lambda x: vectorizedDot(forwardModelMatrices[0], x, obsShape), \
            lambda x: vectorizedDot(forwardModelMatrices[1], x, obsShape), \
            lambda x: vectorizedDot(forwardModelMatrices[2], x, obsShape)], impulseFrame)

        viewFrame(convolvedFrame, adaptiveScaling=True)

    if MACARENA_CORRECT_DECONV:
        inversionMatrices = pickle.load(open("macarena_inverter_gt.p", "r"))

        vid = pickle.load(open("macarena_dark_fixed_cropped_meansub.p", "r"))

    #    print vid.shape

    #    recoveryShape = (42, 23)
        recoveryShape = (52, 33)

        recoveredVid = []

        for i, frame in enumerate(vid):
            print i

            deconvolvedFrame = doSeparateFuncToEachChannelSeparated([lambda x: vectorizedDot(inversionMatrices[0], x, recoveryShape), \
                lambda x: vectorizedDot(inversionMatrices[1], x, recoveryShape), \
                lambda x: vectorizedDot(inversionMatrices[2], x, recoveryShape)], frame)

            deconvolvedFrame = np.sum(deconvolvedFrame, axis=2)/3

            recoveredVid.append(imageify(deconvolvedFrame))
    #    inversionMatrix1 = pickle.load(open("orange_inverter_exp_1.p", "r"))
     #   inversionMatrix2 = pickle.load(open("orange_invert
        convertArrayToVideo(np.array(recoveredVid), 0.5, "macarena_dark_fixed_cropped_meansub_recovery_77", 15) 

    if MACARENA_EXP_DECONV:
    #    inversionMatrices = pickle.load(open("macarena_inverter_exp.p", "r"))
    #    inversionMatrices = pickle.load(open("inverter_exp_36225.p", "r"))
    #    inversionMatrix0 = pickle.load(open("orange_inverter_exp_0.p", "r"))
    #    inversionMatrices = pickle.load(open("bld66_inverter_exp.p", "r"))
        inversionMatrices = pickle.load(open("bld66_jank_inverter_exp.p", "r"))

     #   inversionMatrices = [inversionMatrix0, inversionMatrix1, inversionMatrix2]

    #    vid = pickle.load(open("macarena_dark_fixed_cropped_meansub.p", "r"))
        vid = pickle.load(open("bld66_rect_meansub.p", "r"))
    #    vid = pickle.load(open("orange_rect_meansub_medfilt.p", "r"))

    #    print vid.shape

    #    recoveryShape = (42, 23)
    #    recoveryShape = (52, 33)
     #   recoveryShape = (30, 60)
    #    recoveryShape = (65, 65)
    #    recoveryShape = (39, 70)
        recoveryShape = (25, 45)

        recoveredVid = []

        for i, frame in enumerate(vid):
            print i

            deconvolvedFrame = doSeparateFuncToEachChannelSeparated([lambda x: vectorizedDot(inversionMatrices[0], x, recoveryShape), \
                lambda x: vectorizedDot(inversionMatrices[1], x, recoveryShape), \
                lambda x: vectorizedDot(inversionMatrices[2], x, recoveryShape)], frame)

            deconvolvedFrame = np.sum(deconvolvedFrame, axis=2)/3

            recoveredVid.append(imageify(deconvolvedFrame))

        convertArrayToVideo(np.array(recoveredVid), 1, "bld66_recovery_exp", 15) 
    #    convertArrayToVideo(np.array(recoveredVid), 5, "orange_recovery_exp", 15) 


    if RECOVER_SCENE:
    #    vid = pickle.load(open("macarena_dark_fixed_cropped_diff_medfilt.p", "r"))
    #    vid = pickle.load(open("orange_rect_diff_medfilt.p", "r"))
        vid = pickle.load(open("bld66_rect_diff_medfilt.p", "r"))

        estimateSceneFromDifferenceFramesCanvasPreserving(vid)



    if MESS_AROUND_WITH_OBS:
        vid = pickle.load(open("bld66_rect_diff_medfilt.p", "r"))

        listOfMatchArrays = []

        for i, frame in enumerate(vid):
            print i

            listOfMatchArrays.append(getMatchArrayImage(frame, frame))

    #        viewFrame(getMatchArrayImage(frame, frame), adaptiveScaling=True, differenceImage=True)

    #    viewFrame(vid[390], adaptiveScaling=True, differenceImage=True)

        pickle.dump(np.array(listOfMatchArrays), open("bld66_match.p", "w"))

    if MESS_AROUND_WITH_OBS_2:
        matchArrays = pickle.load(open("bld66_match.p", "r"))

        matchArraySum = np.zeros(matchArrays[0].shape)

        for i, matchArray in enumerate(matchArrays[:300]):
            print i

            if i % 200 == 0:
                viewFrame(matchArraySum, adaptiveScaling=True, differenceImage=True)

    #        viewFrame(matchArray, adaptiveScaling=True, differenceImage=True)

            matchArraySum += matchArray

    #        viewFrame(matchArraySum, adaptiveScaling=True, differenceImage=True)

        viewFrame(matchArraySum, adaptiveScaling=True, differenceImage=True)

    #    convertArrayToVideo(np.array(listOfMatchArrays), 1, "match_arrays", 15) 

    if AUTOCORR_TEST:
        vid = pickle.load(open("steven_batched.p", "r"))

        frame = vid[300]

        autoCorrFrame = getMatchArrayImage(frame, frame)

    #    viewFrame(frame)
    #    viewFrame(autoCorrFrame, adaptiveScaling=True)

        print frame.shape

        averageFramePixel = np.sum(np.sum(frame, 0), 0)

        print averageFramePixel.shape

        summedFrame = np.array([[averageFramePixel]*frame.shape[1]]*frame.shape[0])/(frame.shape[0]*frame.shape[1])
        print summedFrame.shape

        autoCorrSummedFrame = getMatchArrayImage(summedFrame, summedFrame)


    #    viewFrame(summedFrame, adaptiveScaling=True)
    #    viewFrame(autoCorrSummedFrame, adaptiveScaling=True)

        meanSubFrame = frame - summedFrame

    #    viewFrame(meanSubFrame, adaptiveScaling=True, differenceImage=True, magnification=0.5)
        autoCorrMeanSubFrame = getMatchArrayImage(meanSubFrame, meanSubFrame)

        viewFrame(autoCorrMeanSubFrame, adaptiveScaling=True, differenceImage=True, magnification=1)

    if AUTOCORR_TEST_2:

        miFrame = imageify(makeMultipleImpulseFrame((20, 20), 4))

        autoCorrFrame = getMatchArrayImage(miFrame, miFrame)

        viewFrame(miFrame)
        viewFrame(autoCorrFrame, adaptiveScaling=True)

    if JANK_RECOVERY:
        kernelShape = (35, 25)

        offset1 = 15

    #    gauss1 = getGaussianKernelVariableLocation(kernelShape, np.array([35, 10]), 2)
    #    gauss2 = getGaussianKernelVariableLocation(kernelShape, np.array([20, 10]), 2)
    #    gauss3 = getGaussianKernelVariableLocation(kernelShape, np.array([17.5, 20]), 2)
    #    gauss4 = getGaussianKernelVariableLocation(kernelShape, np.array([35, 30]), 2)

        gauss1 = getGaussianKernelVariableLocation(kernelShape, np.array([35-offset1, 10]), 2)
        gauss2 = getGaussianKernelVariableLocation(kernelShape, np.array([20-offset1, 10]), 2)
        gauss3 = getGaussianKernelVariableLocation(kernelShape, np.array([17.5-offset1, 20]), 2)
        gauss4 = getGaussianKernelVariableLocation(kernelShape, np.array([35-offset1, 30]), 2)
        
        jankKernel = gauss1 + gauss2 + gauss3 + gauss4

        viewFrame(imageify(jankKernel))
        
        forwardModelMatrices = doFuncToEachChannelSeparated(lambda x: \
            getForwardModelMatrix2DToeplitzFullFlexibleShape(x, (40,70), (10, 10)), imageify(jankKernel))

    #    for i, forwardModelMatrix in enumerate(forwardModelMatrices):
    #        inversionMatrix = getPseudoInverse(forwardModelMatrix, 1e5) 

    #        pickle.dump(inversionMatrix, open("orange_inverter_exp_" + str(i) + ".p", "w"))

    #    inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1.5e-8) for forwardModelMatrix in forwardModelMatrices]
        inversionMatrices = [getPseudoInverse(forwardModelMatrix, 1e-6) for forwardModelMatrix in forwardModelMatrices]

        pickle.dump(inversionMatrices, open("bld66_jank_inverter_exp.p", "w"))     

    #    jankKernel = np.zeros()

    if CVPR_EXAMPLES:

        imRaw = Image.open("occluder_example.png")
        im = np.array(imRaw).astype(float)

    #    viewFrame(im, adaptiveScaling=True)


        flatIm = np.sum(im, 2)/(3*255) - np.ones(im.shape[:-1])/3

    #    viewFrame(imageify(flatIm))

        resizedIm = resizeArray(flatIm, (100, 95))

        def roundToZeroOne(x):
            if x > 0.5:
                return 1
            else:
                return 0

        thresholdedIm = np.vectorize(roundToZeroOne)(resizedIm)

    #    print resizedIm+

        viewFrame(imageify(thresholdedIm), adaptiveScaling=True)

    #    path = "lindsey.mp4"
    #    vid = imageio.get_reader(path,  'ffmpeg')

    #    frame1a = np.sum(vid.get_data(200).astype(float), 2)
     #   frame1b = np.sum(vid.get_data(201).astype(float), 2)
    #    diffFrame1 = frame1b - frame1a 

    #    diffFrame1 = resizeArray(diffFrame1, (100, 130))

        diffFrame1 = np.array(Image.open("frame1.png")).astype(float)

        flatIm1 = np.sum(diffFrame1, 2)/(3*255) - np.ones(diffFrame1.shape[:-1])/3

        resizedIm1 = resizeArray(flatIm1, (100, 95))
        convFrame1 = convolve2DToeplitzFull(thresholdedIm, resizedIm1)

        viewFrame(imageify(resizedIm1))

        diffFrame2 = np.array(Image.open("frame2.png")).astype(float)

        flatIm2 = np.sum(diffFrame2, 2)/(3*255) - np.ones(diffFrame2.shape[:-1])/3

        resizedIm2 = resizeArray(flatIm2, (100, 95))

        viewFrame(imageify(resizedIm2))

        convFrame2 = convolve2DToeplitzFull(thresholdedIm, resizedIm2)
        viewFrame(imageify(convFrame1), adaptiveScaling=True)
        viewFrame(imageify(convFrame2), adaptiveScaling=True)

        matchArray, bestMatchArray, bestMatchIndex, matchQuality = \
            getMatchArray(convFrame1, convFrame2)

    #        viewFrame(imageify(matchArray), adaptiveScaling=True)

        overlapArray1, overlapArray2 = getOverlapArray(convFrame1, \
            convFrame2, bestMatchIndex)

        viewFrame(imageify(matchArray), adaptiveScaling=True)
        viewFrame(imageify(overlapArray1), adaptiveScaling=True)
        viewFrame(imageify(overlapArray2), adaptiveScaling=True)

        overallOverlapArray = np.multiply(np.sqrt(overlapArray1), np.sqrt(overlapArray2))

        viewFrame(imageify(overallOverlapArray), adaptiveScaling=True)

    if CVPR_COMPUTE_HAMMING_DISTANCE:
        groundTruthOcc = pickle.load(open("corr_occ_2.p", "r"))
        groundTruthOccThresholded = np.vectorize(lambda x: 1*(x>0.5))(groundTruthOcc)
        phaseRetrieveOcc = pickle.load(open("phase_retrieve_cvpr.p", "r"))
        phaseRetrieveOccThresholded = np.vectorize(lambda x: 1*(x>0.5))(phaseRetrieveOcc)
        binaryOcc = pickle.load(open("extracted_occ_bin.p", "r"))
        binaryOccThresholded = np.vectorize(lambda x: 1*(x>0.5))(binaryOcc)


        viewFrame(imageify(groundTruthOcc))
        viewFrame(imageify(phaseRetrieveOccThresholded))
        viewFrame(imageify(binaryOcc))

        print groundTruthOccThresholded


        print hammingDistance(groundTruthOccThresholded, phaseRetrieveOccThresholded)
        print hammingDistance(groundTruthOccThresholded, binaryOccThresholded)
        print np.shape(groundTruthOcc)[0]*np.shape(groundTruthOcc)[1]

    if CVPR_MAKE_EXAMPLE_MOVIE:

        diffVid = pickle.load(open("steven_batched_diff.p", "r"))

        vid = pickle.load(open("steven_batched.p", "r"))

    #    occ = pickle.load(open())

        for i in range(len(diffVid)):
            print i

            frame = vid[i]
            diffFrame = diffVid[i]

            p.clf()

            p.subplot(121)
            p.axis("off")
            viewFrame(frame, filename="pass", adaptiveScaling=False,
                relax=True, magnification=1, differenceImage=False)

            p.subplot(122)
            p.axis("off")
            viewFrame(diffFrame, filename="pass", adaptiveScaling=True,
                relax=True, magnification=1, differenceImage=True)

            p.savefig("blind_deconv_movie_sim_2/frame_" + padIntegerWithZeros(i, 4) + ".png")

    if CVPR_MAKE_EXAMPLE_MOVIE_2:

        diffVid = pickle.load(open("steven_batched_diff.p", "r"))

        vid = pickle.load(open("steven_batched.p", "r"))

        occ = pickle.load(open("corr_occ_2.p", "r"))

        for i in range(len(diffVid)):
            print i

            frame = vid[i]
            diffFrame = diffVid[i]

            obsFrame = addNoise(doFuncToEachChannel(lambda x: convolve2DToeplitzFull(x, occ), diffFrame))

            p.clf()

            viewFrame(obsFrame, filename="pass", adaptiveScaling=True,
                relax=True, magnification=1, differenceImage=True)

            p.savefig("blind_deconv_movie_sim_3/frame_" + padIntegerWithZeros(i, 4) + ".png")