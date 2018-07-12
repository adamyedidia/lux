from __future__ import division
import numpy as np
import random
from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
from image_distortion_simulator import imageify, imageifyComplex
from scipy.signal import convolve, deconvolve
from numpy.fft import fft, ifft
from math import sqrt, pi, exp, log, sin, cos
from import_1dify import batchList, displayConcatenatedArray
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

LOOK_AT_FREQUENCY_PROFILES = False
DIVIDE_OUT_STRATEGY = False
POLYNOMIAL_STRATEGY = False
UNDO_TRUNCATION = False
BILL_NOISY_POLYNOMIALS = True
DENSITY_TESTS = False

ZERO_DENSITY = 2
NONZERO_DENSITY = 20
FLIP_DENSITY = 10
SIGNAL_SIGMA = 1
NOISE_SIGMA = 1

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

class LogLaplaceScale(mscale.ScaleBase):
    name = 'log_laplace'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The MercatorLatitudeTransform class is defined below as a
        nested class of this one.
        """
        return self.LogLaplaceTransform()

    class LogLaplaceTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
#            return np.power(a, 0.5)
            return exp(abs(log(a)))

        def inverted(self):
            return LogLaplaceScale.InvertedQuadraticTransform()

    def set_default_locators_and_formatters(self, axis):
        pass

    class InvertedQuadraticTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return np.power(a, 2)

        def inverted(self):
            return QuadraticScale.QuadraticTransform()


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

def loglaplace(x, mu, sigma):
    return exp(-abs(log(x-mu))/(sigma))/(2*x*sigma)
    
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

def visualizeColorMesh(X, Y, Z, roots):
#    p.pcolormesh(X, Y, Z, cmap=cm.gnuplot2)
    p.pcolormesh(X, Y, Z, cmap=cm.prism)
    
    p.colorbar()
    
    for root in roots:
        p.plot(np.real(root), np.imag(root), "w.")
    
    unitCircle = p.Circle((0, 0), 1, color='k', fill=False)    
        
    ax = p.gca()
        
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])    
    ax.set_aspect("equal")
    ax.add_artist(unitCircle) 
    
    p.show()

def visualizePolynomialValues(seq, numSteps=300, minX=-1.5, maxX=1.5,
     minY=-1.5, maxY=1.5):

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
        
        
        
    
    Z = np.log(np.divide(np.abs(np.polyval(seq, X + 1j*Y)), \
        np.abs(np.polyval(seq, np.exp(2*pi*1j*random.random())*np.abs(X + 1j*Y)))))
        
        
        
#    Z = X + 1j*Y
            
    visualizeColorMesh(X, Y, Z, np.roots(seq))
    visualizeColorMesh(X, Y, Z, [])
    
def makePolynomialMesh(seq, numSteps=300, minX=-1.5, maxX=1.5,
     minY=-1.5, maxY=1.5):  
    
    xRange = np.linspace(minX, maxX, numSteps)
    yRange = np.linspace(minY, maxY, numSteps)
        
    X, Y = np.meshgrid(xRange, yRange)    

    Z = np.log(np.divide(np.abs(np.polyval(seq, X + 1j*Y)), \
        np.abs(np.polyval(seq, np.exp(2*pi*1j*random.random())*np.abs(X + 1j*Y))))+1)
 
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
 
    p.plot(xRange, [loglaplace(i, 0, sigma) for i in xRange])  
  
    p.axvline(x=1, color="k")
    p.show()
    
        
    
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
    
def makeAggregateMesh(listOfSeqs, numSteps=300, minX=-1.5, maxX=1.5,
    minY=-1.5, maxY=1.5):
    
    averageZ = 0
    
    for i, seq in enumerate(listOfSeqs):
        
        if i % 100 == 0:
            pront(str(i) + "/" + str(len(listOfSeqs)))
            
        
        X, Y, Z = makePolynomialMesh(seq, numSteps=300, minX=-1.5, maxX=1.5,\
            minY=-1.5, maxY=1.5)    
            
        averageZ += Z
        
    averageZ /= len(listOfSeqs)
    
    return X, Y, averageZ
#    for _ in range()    

def makeRadialMesh(seq, densityFunc, thetaSteps=1000, rSteps=1000):
    
    thetaRange = np.linspace(0, 2*pi, thetaSteps)    
    
    rPointsParam = rSteps/10
    
    rRange = makeEvenPointsAccordingToDensity(densityFunc, rPointsParam, 0, 10, 1, \
        rSteps*2)

    THETA, R = np.meshgrid(thetaRange, rRange)
    
#    print R, THETA
    
    Z = np.log(np.divide(np.abs(np.polyval(seq, np.multiply(np.exp(1j*THETA), R))), \
        np.abs(np.polyval(seq, np.exp(2*pi*1j*random.random())*R)))+1)

    rProxyRange = np.linspace(0, 1, len(rRange))    
    THETA_PROXY, R_PROXY = np.meshgrid(thetaRange, rProxyRange)

    return THETA_PROXY, R_PROXY, Z

    
def makeAggregateRadialMesh(listOfSeqs, densityFunc, thetaSteps=1000, rSteps=1000):
    
    averageZ = 0
    
    for i, seq in enumerate(listOfSeqs):
        
        if i % 100 == 0:
            pront(str(i) + "/" + str(len(listOfSeqs)))
            
        
        THETA, R, Z = makeRadialMesh(seq, densityFunc, thetaSteps, rSteps)        
            
        averageZ += Z
        
    averageZ /= len(listOfSeqs)
    
    return THETA, R, averageZ
     
def visualizeRadialColorMesh(THETA, R, Z, roots):
    
    print Z.shape
    
    p.pcolormesh(THETA, R, Z, cmap=cm.prism)

    p.colorbar()
    
    for root in roots:
        p.plot(np.angle(root, deg=False), np.abs(root), "k.")
        
    p.axhline(y=1, color="k")
    
    ax = p.gca()
    
#    ax.set_xlim([0, 2*pi])
#    ax.set_ylim([0, 10])

    ax.set_aspect(0.5)

#    ax.set_yscale("log_laplace")

    p.show()    
    
    
    
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
        
#    listOfFlatFrames = batchList(pickle.load(open("flat_frames_fine.p", "r")), 2)
    listOfFlatFrames = batchList(pickle.load(open("flat_frames.p", "r")), 2)


    n = len(listOfFlatFrames[0])

    pront("n = " + str(n))

    listOfFlatDifferenceFrames = [listOfFlatFrames[i+1] - listOfFlatFrames[i] \
        for i in range(len(listOfFlatFrames) - 1)]

#    occluder = generateZeroOneSeq(2*n-1)    
    occluder = np.array([1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,1,0])
            
#    viewFlatFrame(imageify(occluder))

    occluderPolynomial = Polynomial(occluder[::-1])

#    displayRoots(occluderPolynomial, [], 0, 1)

    listOfConvolvedDifferenceFrames = [addNoise(doFuncToEachChannelVec(convolveMaker(occluder), frame)) for frame in listOfFlatDifferenceFrames]

    concatenatedDifferenceFrames = np.concatenate(listOfFlatDifferenceFrames, 1)
    convolvedDifferenceFrames = np.concatenate(listOfConvolvedDifferenceFrames, 1)
    
    
#    displayConcatenatedArray(concatenatedDifferenceFrames, magnification=100, \
#        differenceImage=True, rowsPerFrame=1)
    
#    displayConcatenatedArray(convolvedDifferenceFrames, magnification=100, \
#        differenceImage=True, rowsPerFrame=1, stretchFactor=5)    

    solveProblem = True
    singlePoly = False

    densitySigma = 0.3/sqrt(n)
    densityFunc = lambda x: loglaplace(x, 0, densitySigma) 

    

    if solveProblem: 

        listOfSingleColorFrames = getListOfSingleColorFrames(listOfConvolvedDifferenceFrames)[:300]

#        makeRootMagnitudeHistogram(listOfSingleColorFrames, 0.3/sqrt(n))



        THETA, R, Z = makeAggregateRadialMesh(listOfSingleColorFrames, densityFunc)
        
        visualizeRadialColorMesh(THETA, R, Z, np.roots(occluder))
        visualizeRadialColorMesh(THETA, R, Z, [])
        
        X, Y, Z = makeAggregateMesh(listOfSingleColorFrames)
        
        visualizeColorMesh(X, Y, Z, np.roots(occluder))
        visualizeColorMesh(X, Y, Z, [])        
            

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
    densityFunc = lambda x: loglaplace(x, 0, sigma)
    
    numPointsParam = 10
    
#    evenPoints = makeEvenPointsAccordingToDensity(densityFunc, numPointsParam, 0, \
#        100, 1, 30)
        
    evenPoints = makeEvenPointsAccordingToDensity(densityFunc, 10*numPointsParam, 0, \
        100, 1, 300)   