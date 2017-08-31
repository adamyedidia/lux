from __future__ import division
from theoretical_limits import getTransferMatrixFromOccluder, getTransferMatrixFrom2DOccluder, \
    createSquare2DOccluder, createCircle2DOccluder, createDiamond2DOccluder
import numpy as np
from math import log, sqrt, pi
from search import randomBits
import matplotlib.pyplot as p

def factorVsN(getOccluderFunc, maxN=1001):
    listOfFactors = []
    for n in range(2, maxN+1):
        occLen = 2*n-1
        occluder = getOccluderFunc(occLen)
        transferMatrix = getTransferMatrixFromOccluder(occluder, n)
        logDet = np.linalg.slogdet(transferMatrix)[1]
        factor = 20*logDet / (n * log(n, 2))

        listOfFactors.append(factor)

        print n, factor

    p.plot(listOfFactors)
    ax = p.gca()
    ax.set_xlabel("matrix size")
    ax.set_ylabel("20 logdet / n log n (aka decibels per pixel??)")
    p.show()

def factorVsN2D(getOccluderFunc, maxSL=101):
    listOfFactors = []
    listOfNs = []

    for sl in range(3, maxSL+1, 2):
        occluder = getOccluderFunc(sl)
#        p.matshow(occluder)
#        p.show()
        windowSL = int((sl+1)/2)
        transferMatrix = getTransferMatrixFrom2DOccluder(occluder, (windowSL, \
            windowSL))
        fullLogDet = np.linalg.slogdet(transferMatrix)
        logDet = max(fullLogDet[1], 0)
        n = windowSL*windowSL
        factor = 20*logDet / (n * log(n, 2))

        listOfFactors.append(factor)
        listOfNs.append(n)

        print n, factor

    p.plot(listOfNs, listOfFactors)
    ax = p.gca()
    ax.set_xlabel("matrix size")
    ax.set_ylabel("20 logdet / n log n (aka decibels per pixel??)")
    p.show()

def identityOccluderFunc(occLen):
    return [0] * int((occLen-1)/2) + [1] + [0] * int((occLen-1)/2)

def randomOccluderFunc(occLen):
    return randomBits(occLen)

def createSquare2DOccluderQuarterArea(sl):
    return createSquare2DOccluder((sl, sl), sl/2)

def createCircle2DOccluderQuarterArea(sl):
    return createCircle2DOccluder((sl, sl), sl/sqrt(4*pi))

def createDiamond2DOccluderQuarterArea(sl):
    return createDiamond2DOccluder((sl, sl), sl/2-2)

factorVsN2D(createCircle2DOccluderQuarterArea)
