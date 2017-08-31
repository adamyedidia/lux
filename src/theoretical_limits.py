from __future__ import division
import numpy as np
from math import log, sqrt, pi
import matplotlib.pyplot as p
import pickle
from search import randomBits, randomGreedySearch


def findMutualInfo(A, sigma):
    ata = np.dot(np.transpose(A), A)
    n = ata.shape[0]
    return np.linalg.slogdet(ata/(sigma*sigma) + np.identity(n))[1] / log(2)

def getTransferMatrixFromOccluder(occluderArray, slidingWindowSize):
    transferMatrix = []

    for i in range(len(occluderArray) - slidingWindowSize, -1, -1):
        transferMatrix.append(occluderArray[i:i+slidingWindowSize])

    return np.array(transferMatrix)

def getUpperBound(n, sigma):
    return n/2*log(n,2) - n + 1 + n*log(n/(sigma*sigma)+1, 2)

def getDetOfToepMaker(n):
    def getDetOfToep(l):
        return np.linalg.det(getTransferMatrixFromOccluder(l, n))
    return getDetOfToep

def getTransferMatrixFrom2DOccluder(occluderArray, slidingWindowShape):
    transferMatrix = []

    for i in range(occluderArray.shape[0] - slidingWindowShape[0], -1, -1):
        for j in range(occluderArray.shape[1] - slidingWindowShape[1], -1, -1):
            subWindow = occluderArray[i:i+slidingWindowShape[0],j:j+slidingWindowShape[1]]
            transferMatrix.append(subWindow.flatten())

    return np.array(transferMatrix)

def createCircle2DOccluder(shape, radius):
    occluderArray = []

    for i in range(shape[0]):
        occluderArray.append([])
        for j in range(shape[1]):
            if sqrt((i - (shape[0]-1)/2)**2 + (j - (shape[1]-1)/2)**2) <= radius:
                occluderArray[-1].append(1)
            else:
                occluderArray[-1].append(0)

    return np.array(occluderArray)

def createSquare2DOccluder(shape, sideLength):
    occluderArray = []

    for i in range(shape[0]):
        occluderArray.append([])
        for j in range(shape[1]):
            if (abs(i - (shape[0]-1)/2) <= sideLength/2) and (abs(j - (shape[1]-1)/2) <= sideLength/2):
                occluderArray[-1].append(1)
            else:
                occluderArray[-1].append(0)

    return np.array(occluderArray)

def createDiamond2DOccluder(shape, sideLength):
    occluderArray = []

    for i in range(shape[0]):
        occluderArray.append([])
        for j in range(shape[1]):
            if (abs(i + j - (shape[0]-1)/2 - (shape[1]-1)/2) <= sideLength*sqrt(2)/2) and \
                (abs(i - j - (shape[0]-1)/2 + (shape[1]-1)/2) <= sideLength*sqrt(2)/2):
                occluderArray[-1].append(1)
            else:
                occluderArray[-1].append(0)

    return np.array(occluderArray)

if False:

    n = 101
    #n = 21
    edgeOccluder = [1]*n + [0]*(n-1)
    edgeTM = getTransferMatrixFromOccluder(edgeOccluder, n)
    pinholeTM = np.identity(n)
    bestMatList = randomGreedySearch(randomBits(2*n-1), getDetOfToepMaker(n), "max",
        6*n)
    bestMat = getTransferMatrixFromOccluder(bestMatList, n)
    #bestMat = np.identity(n)
    randomMat = getTransferMatrixFromOccluder(randomBits(2*n-1), n)
    #p.matshow(bestMat)
    #p.show()
    #print bestMat

    area = 2500
    shape = (101, 101)
    innerShape = (51, 51)

    circleOccluder = createCircle2DOccluder(shape, sqrt(area/pi))
    squareOccluder = createSquare2DOccluder(shape, sqrt(area))
    diamondOccluder = createDiamond2DOccluder(shape, sqrt(area))

    circleTransferMat = getTransferMatrixFrom2DOccluder(circleOccluder, innerShape)
    squareTransferMat = getTransferMatrixFrom2DOccluder(squareOccluder, innerShape)
    diamondTransferMat = getTransferMatrixFrom2DOccluder(diamondOccluder, innerShape)
    #p.matshow(circleTransferMat)
    #p.show()
    #p.matshow(squareTransferMat)
    #p.show()
    #p.matshow(diamondTransferMat)
    #p.show()

    #p.matshow(circleOccluder)
    #p.show()
    #p.matshow(squareOccluder)
    #p.show()
    #p.matshow(diamondOccluder)
    #p.show()

    #bestMat = getTransferMatrixFromOccluder(pickle.load(open("bestlist.p", "r")), n)
    #sigma = 11

    listOfLogSigmas = []
    listOfEdgeMutualInfos = []
    listOfPinholeMutualInfos = []
    listOfBestMutualInfos = []
    listOfRandomMutualInfos = []
    listOfCircleMutualInfos = []
    listOfSquareMutualInfos = []
    listOfDiamondMutualInfos = []
    listOfUpperBounds = []
    listOfDBs = []


    for logSigma in np.linspace(-2.5, 2.5, 100):
        sigma = 10**logSigma
        listOfLogSigmas.append(logSigma)
        listOfDBs.append(logSigma*-20)
        listOfEdgeMutualInfos.append(findMutualInfo(edgeTM, n, sigma)/n)
        listOfPinholeMutualInfos.append(findMutualInfo(pinholeTM, n, sigma)/n)
        listOfBestMutualInfos.append(findMutualInfo(bestMat, n, sigma)/n)
        listOfRandomMutualInfos.append(findMutualInfo(randomMat, n, sigma)/n)
    #    listOfCircleMutualInfos.append(findMutualInfo(circleTransferMat, n, sigma)/n)
    #    listOfSquareMutualInfos.append(findMutualInfo(squareTransferMat, n, sigma)/n)
    #    listOfDiamondMutualInfos.append(findMutualInfo(diamondTransferMat, n, sigma)/n)
        listOfUpperBounds.append(getUpperBound(n, sigma)/n)

    ax = p.gca()
    #ax.set_xlabel("Log(sigma)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bits per pixel")

    p.plot(listOfDBs, listOfEdgeMutualInfos, "b-")
    p.plot(listOfDBs, listOfPinholeMutualInfos, "g-")
    p.plot(listOfDBs, listOfBestMutualInfos, "c-")
    p.plot(listOfDBs, listOfUpperBounds, "r-")
    #p.plot(listOfDBs, listOfCircleMutualInfos, "m-")
    #p.plot(listOfDBs, listOfSquareMutualInfos, "y-")
    #p.plot(listOfDBs, listOfDiamondMutualInfos, "b-")
    p.plot(listOfDBs, listOfRandomMutualInfos, "k-")

    p.show()
    #print findMutualInfo(A, n, sigma)
