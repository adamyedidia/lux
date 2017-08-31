from __future__ import division
import numpy as np
from math import log, sqrt, pi
import matplotlib.pyplot as p
import pickle
from search import randomBits, randomGreedySearch, exhaustiveSearch
from theoretical_limits import findMutualInfo, getTransferMatrixFromOccluder

#MAT_SIDE_LENGTH = 500
#DEPTH = 50


def qualityTestMaker(n, sigma):
    def qualityTest(occluderArray):
        A = getTransferMatrixFromOccluder(occluderArray, n)
        return findMutualInfo(A, sigma)
    return qualityTest

n = 7
occluderArraySize = 2*n-1

outputMat = []

for logSigma in np.linspace(-4, 2, 30):
    sigma = 10**logSigma
    print sigma

    qualityTest = qualityTestMaker(n, sigma)

#    occluderArray = randomGreedySearch(randomBits(occluderArraySize), qualityTest, \
#        "max", 10*n)

    occluderArray = exhaustiveSearch(occluderArraySize, qualityTest, "max")

    outputMat.append(occluderArray)

ax = p.gca()
ax.set_xlabel("Occluder X position")
ax.set_ylabel("Log Sigma")
ax.imshow(outputMat, interpolation="none", extent=[0, 2*n-1, 2, -4], aspect=36/n)

#p.matshow(outputMat)
p.show()
