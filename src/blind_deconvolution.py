from __future__ import division
import numpy as np
import random
from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
from image_distortion_simulator import imageify, imageifyComplex
from scipy.signal import convolve, deconvolve, argrelextrema
from numpy.fft import fft, ifft
from math import sqrt, pi, exp, log, sin, cos, floor, ceil
from import_1dify import batchList, displayConcatenatedArray, fuzzyLookup
from image_distortion_simulator import doFuncToEachChannelVec
import pickle
from numpy.polynomial.polynomial import Polynomial, polyfromroots
import matplotlib.pyplot as p
from scipy.linalg import circulant as circ
import matplotlib.cm as cm 
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator
from matplotlib import rcParams
from custom_plot import createCMapDictHelix, LogLaplaceScale
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import broyden1, fmin_bfgs, fmin_cg, fmin_tnc
from pynverse import inversefunc
from scipy.integrate import quad

LOOK_AT_FREQUENCY_PROFILES = False
DIVIDE_OUT_STRATEGY = False
POLYNOMIAL_STRATEGY = False
UNDO_TRUNCATION = False
BILL_NOISY_POLYNOMIALS = False
DENSITY_TESTS = False
GIANT_GRADIENT_DESCENT = False
ROADSIDE_DECONVOLUTION = True

ZERO_DENSITY = 2
NONZERO_DENSITY = 20
FLIP_DENSITY = 10
SIGNAL_SIGMA = 1
NOISE_SIGMA = 1e-2

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

    return sum(occGradContributionFromSingleElt(convolvedElt, groundTruthElt, lambda6)) for \
        convolvedElt, groundTruthElt, in zip(convolvedFrame, groundTruthFrame)

def getGradientFromObservationMatching(convolvedFrames, groundTruth, lambda6, 
    occIndex, getCoefficient, hypothesizedFrame):

    convolvedFrames = [getConvolvedFrame(occluderSeq, hypothesizedFrame) for hypothesizedFrame in listOfFrames]

    gradientContribution = [sum([getGradientContributionFromSingleFrame(convolvedFrame, groundTruthFrame, 
        lambda6, occIndex, getCoefficient, hypothesizedFrame) for \
        convolvedFrame, groundTruthFrame in zip(convolvedFrames, groundTruth)]) for occIndex in range(occLength)]


    return np.array(gradientContribution)


def getOccluderGradient(convolvedFrames, groundTruth, occIndex, getCoefficient, hypothesizedFrame, 
    lambda1, lambda2, lambda6):

    gradientFromObservationMatching = getGradientFromObservationMatching(convolvedFrames, groundTruth, lambda6, 
        occIndex, getCoefficient, hypothesizedFrame)

    gradientFromSparsity = occGradSparsity(occVal, lambda1)
    gradientFromSpatial = occGradSpatial(occVal, lambda2)

    return gradientFromObservationMatching + gradientFromSparsity + gradientFromSpatial




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
    lambda1 = -0.3

    awayFromUnitCircle = lambda1 * sqrt(abs(abs(x) - 1))

    lambda2 = -0.3 

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
    print point

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

            print thetaGridPoint, warpR(rGridPoint, evenPoints)

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

    normalizationAngle = getNormalizationAngleDistant(listOfSeqs)
#    normalizationAngle = getNormalizationAngleRandom(listOfSeqs)

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
    visualizeColorMesh(X, Y, Z, correctRoots, normalizationAngle=normalizationAngle)

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

    print sortedFoundRoots

    if len(sortedFoundRoots) > numRootsExpected:
        recoveredRoots = [i[0] for i in sortedFoundRoots[:numRootsExpected]]
    else:
        recoveredRoots = [i[0] for i in sortedFoundRoots]

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

def findRootsExhaustive():
    pass


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
    
    
    
    displayConcatenatedArray(concatenatedDifferenceFrames, magnification=100, \
        differenceImage=True, rowsPerFrame=1)
    
    displayConcatenatedArray(convolvedDifferenceFrames, magnification=100, \
        differenceImage=True, rowsPerFrame=1)    
    
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
        
    listOfFlatFrames = batchList(pickle.load(open("flat_frames_fine.p", "r")), 2)
#    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)
#    listOfFlatFrames = batchList(pickle.load(open("flat_frames_coarse.p", "r")), 2)


    n = len(listOfFlatFrames[0])

    pront("n = " + str(n))

    listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
        for i in range(len(listOfFlatFrames) - 1)]

#    occluder = generateZeroOneSeq(2*n-1)    
    occluder = np.array([1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1])
            
#    occluder = np.array([1,0,0,1,1])

#    print polyfromroots(np.roots(occluder))

    print np.roots(occluder)

#    viewFlatFrame(imageify(occluder))

    occluderPolynomial = Polynomial(occluder[::-1])

#    displayRoots(occluderPolynomial, [], 0, 1)

    listOfConvolvedDifferenceFrames = [addNoise(doFuncToEachChannelVec(convolveMaker(occluder), frame)) \
        for frame in listOfFlatDifferenceFrames]

    concatenatedDifferenceFrames = np.concatenate(listOfFlatDifferenceFrames, 1)
    convolvedDifferenceFrames = np.concatenate(listOfConvolvedDifferenceFrames, 1)
    
    
    displayConcatenatedArray(concatenatedDifferenceFrames, magnification=100, \
        differenceImage=True, rowsPerFrame=10)
    
    displayConcatenatedArray(convolvedDifferenceFrames, magnification=100, \
        differenceImage=True, rowsPerFrame=10, stretchFactor=1)

    solveProblem = True
    singlePoly = False
    truncation = False

    densitySigma = 0.3/sqrt(n)
  
    densityFunc = lambda x: logLaplace(x, 0, densitySigma) 

    

    if solveProblem: 

        listOfSingleColorFrames = getListOfSingleColorFrames(listOfConvolvedDifferenceFrames)[:300]

#        makeRootMagnitudeHistogram(listOfSingleColorFrames, 0.3/sqrt(n))

#        rootsFound = findRoots(listOfSingleColorFrames, np.roots(occluder), showy=True)

        rootsFound = findRoots(listOfSingleColorFrames, np.roots(occluder), \
            numRootsExpected=len(occluder)-1, showy=False)
 
        print rootsFound

        complexFormRootsFound = [root[0] + 1j*root[1] for root in rootsFound]

        print "correct roots", np.roots(occluder)
        print "recovered roots", complexFormRootsFound
        print "correct polynomial", polyfromroots(np.roots(occluder))
        print "recovered polynomial", polyfromroots(complexFormRootsFound)

#        print rootsFound

#        complexFormRootsFound = [np.exp(1j*root[0])*root[1] for root in rootsFound]

        fitPoly = polyfromroots(complexFormRootsFound)
        
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

    listOfFlatFrames = pickle.load(open("flat_frames_grey_bar_obs.p", "w"))

    listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
        for i in range(len(listOfFlatFrames) - 1)]

    
