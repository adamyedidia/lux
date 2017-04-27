from __future__ import division
import numpy as np
import sys
from scipy.linalg import circulant
import matplotlib.pyplot as p
from search import randomBits, randomGreedySearch
from math import log

#n = int(sys.argv[1])

REAL_VALS = [1, 1, 2, 3, 4, 9, 32, 45, 95, 275, 1458, 2240, 6561, 19952, 131072, \
    214245, 755829, 2994003, 19531250, 37579575, 134534444, 577397064, 4353564672, \
    10757577600, 31495183733, 154611524732, 738139162166, 3124126889325, 11937232425585, \
    65455857159975]

EXHAUSTIVE = False
GREEDY = True

def getDeterminantOfCirc(l):
    mat = circulant(l)
#    print mat
    return np.linalg.det(mat)

def allListsOfSizeX(x):
    if x == 0:
        return [[]]

    else:
        oneLess = allListsOfSizeX(x-1)
        return [i + [0] for i in oneLess] + [i + [1] for i in oneLess]

def getLineMatrix(l):
    assert len(l) % 2 == 1
    sideLength = int((len(l) + 1) / 2)
    matArray = []

    for i in range(sideLength):
        matArray.append(l[i:i+sideLength])

    return np.array(matArray)

def getMaxLineMatrix(x):
    bestDet = -float("Inf")
    bestMats = []
    bestLists = []

    allLists = allListsOfSizeX(x)

    for l in allListsOfSizeX(x):

        print l

        mat = getLineMatrix(l)
        det = int(abs(np.linalg.det(mat))+0.001)
        print "det", det

        if det > bestDet:
            bestDet = det
            bestMats = [mat]
            bestLists = [l]
            print "better"

        elif det == bestDet:
            bestMats.append(mat)
            bestLists.append(l)
            print "tied"

    return bestMats, bestLists, bestDet

def inflateList(x, newLength):
    return [x[int(i/newLength*len(x))] for i in range(newLength)]

def getMaxCircMatrix(x):
    bestDet = -float("Inf")
    bestMats = []
    bestLists = []
    allLists = allListsOfSizeX(x)

    for l in allListsOfSizeX(x):
        print l

        mat = circulant(l)
        det = int(abs(np.linalg.det(mat))+0.001)
        print "det", det
        print "bestDet", bestDet

        if det > bestDet:
            bestDet = det
            bestMats = [mat]
            bestLists = [l]
            print "better"

        elif det == bestDet:
            bestMats.append(mat)
            bestLists.append(l)
            print "tied"

    return bestMats, bestLists, bestDet

if EXHAUSTIVE:

    MAT_SIDE_LENGTH = 500
    listOfWinners = []

    MAX_NUM = 16

    for n in range(1,MAX_NUM+1):

        bestMats, bestLists, bestDet = getMaxCircMatrix(n)
        for bestMat in bestMats:
            print bestMat
        for bestList in bestLists:
            print bestList
        print bestDet

        listOfWinners.append(inflateList(bestLists[0], MAT_SIDE_LENGTH))

    arr = np.array(inflateList(listOfWinners, MAT_SIDE_LENGTH))

    print arr
    p.matshow(arr)



    p.show()
    print inflateList([0,1,0], MAT_SIDE_LENGTH)

if GREEDY:

    MAX_NUM = int(sys.argv[1])

    listOfBests = []

    for n in range(1, MAX_NUM+1):
        bestList = randomGreedySearch(randomBits(n), getDeterminantOfCirc, \
            "max", 3*n)

        bestVal = getDeterminantOfCirc(bestList)
        if bestVal > 1:
            logBestVal = log(bestVal)
        else:
            logBestVal = 0

        listOfBests.append(logBestVal)

    p.plot(range(1, MAX_NUM+1), listOfBests)
    p.plot(range(1, MAX_NUM+1), [i/3*log(i) for i in range(1, MAX_NUM+1)])
    p.plot(range(1, len(REAL_VALS)+1), [log(i) for i in REAL_VALS])

    p.show()
