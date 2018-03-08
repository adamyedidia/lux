from __future__ import division
import numpy as np
import sys
from scipy.linalg import circulant
import matplotlib.pyplot as p
from search import randomBits, randomGreedySearch
from math import log, sqrt, pi, floor, exp, cos, sin
import pickle
import string
from scipy.linalg import dft
import random


#n = int(sys.argv[1])

REAL_VALS = [1, 1, 2, 3, 4, 9, 32, 45, 95, 275, 1458, 2240, 6561, 19952, 131072, \
    214245, 755829, 2994003, 19531250, 37579575, 134534444, 577397064, 4353564672, \
    10757577600, 31495183733, 154611524732, 738139162166, 3124126889325, 11937232425585, \
    65455857159975]

BRENT_LIST = False
EXHAUSTIVE = False
GREEDY = False
ARTIFICIAL_MLS = False
BASIC_TEST = False
EXPLORE_LARGE_MLS = False
MATRIX_EIG_ANALYSIS = False
MCMC_SEARCH = False
EIG_ANALYSIS_2 = False
EIG_ANALYSIS_SEARCH = False
EIG_EQUALITY_RESTRICTION = True

def logHBE(n):
    if n % 4 == 0:
        return n/2 * log(n)

    elif n % 2 == 0:
        return log(2) + log(n-1) + (n-2)/2 * log(n-2)

    else:
        return 1/2 * log(2*n-1) + (n-1)/2 * log(n-1)

def logU01(n):
    return logHBE(n+1) - n*log(2)

def testIfMaximal(l):
    mat = circulant(l)

    logdet = np.linalg.slogdet(mat)[1]

    if abs(logdet - logU01(len(l))) < 1e-7:
        return True

    else:
        return False

def assembleFirstRow(groups, valueAssignment):
    n = sum([len(g) for g in groups])

    firstRow = [0] * n

    for i, group in enumerate(groups):
        for j in group:
            firstRow[j] = valueAssignment[i]

    return firstRow

def findGoodGroup(groups, verbose=False, giveUpThreshold=12):
    if len(groups) > giveUpThreshold:
        return (False, None)

    allLists = allListsOfSizeX(len(groups))
    groupLens = [len(g) for g in groups]
    n = sum(groupLens)

    for binList in allLists:
        if dot(binList, groupLens) == floor((n+1)/2):
            if verbose:
#                print "Plausible partition found."
                pass

            firstRow = assembleFirstRow(groups, binList)
            if testIfMaximal(firstRow):
                if verbose:
                    print "Maximal partition found!"
                    print firstRow
                    print binList, groupLens

                return (True, firstRow, binList)

    return (False, None)

def findGoodGroupEfficient(groups):
    if len(groups) > giveUpThreshold:
        return (False, None)

    groupLens = [len(g) for g in groups]
    n = sum(groupLens)
    currentValueAssignment = [0] * n
    i = 0
    currentTotalSize = 0
    maxTotalSize = floor((n+1)/2)
#    minI = 0

    seenZero = False

    while True:
        amountAtStake = groupLens[i]
        if currentValueAssignment[i] == 0:
            seenZero = True

            if currentTotalSize + amountAtStake < maxTotalSize:
                currentValueAssignment[i] = 1
                i += 1
                currentTotalSize += amountAtStake

            else:
#                if i - 1 < minI:
#                    i += 1
#                    minI += 1
                i -= 1

        if currentValueAssignment[i] == 1:
            currentValueAssignment[i] = 0
            i += 1
            currentTotalSize -= amountAtStake


def getDeterminantOfCirc(l):
    mat = circulant(l)
#    print mat
    return np.linalg.det(mat)

def getDeterminantofToeplitz(l):
    mat = circulant(l)

def dot(l1, l2):
    return sum([i*j for i, j in zip(l1, l2)])

def isPrime(n):
    for i in range(2, int(floor(sqrt(n)))+1):
        if n % i == 0:
            return False

    else:
        return True

# take binary list, swap two random non-equal elements
def getRandomSwap(l):
#    print l

    i1 = random.randint(0, len(l)-1)
    i2 = random.randint(0, len(l)-1)

    if l[i1] == l[i2]:
        return getRandomSwap(l)
    else:
        newL = l[:]
        val1 = l[i1]
        val2 = l[i2]
        newL[i1] = val2
        newL[i2] = val1
        return newL

def stepMaybe(evalFunc, l1, l2):
    v1 = evalFunc(l1)
    v2 = evalFunc(l2)

    if random.random() < v2 / (v1 + v2):
        return l2
    else:
        return l1

def stepMaybeEfficient(evalFunc, v1, l2):
    v2 = evalFunc(l2)

    if random.random() < v2 / (v1 + v2):
        return True
    else:
        return False

def stepMaybeEfficientExp(evalFunc, v1, l2):
    v2 = evalFunc(l2)

    v1v2 = np.logaddexp(v1, v2)

    if random.random() < exp(v2 - v1v2):
        return (True, v2)
    else:
        return (False, None)

def stepMaybeEfficientExpGreg(evalFunc, v1, l2):
    v2 = evalFunc(l2)

    if v2 > v1:
#        print "a"

        return (True, v2)

    else:
#        print exp(v2 - v1)

        if random.random() > exp(v2 - v1):
#            print "b"

            return (False, v1)

        else:
#            print "c"

            return (True, v2)


    if random.random() < exp(v2 - v1v2):
        return (True, v2)
    else:
        return (False, None)

def initFuncFloorNPlusOneOver2(n):
    k = int(floor(n+1)/2)
    returnList = [0]*n
    ones = random.sample(range(n), k)
    for i in ones:
        returnList[i] = 1
    return returnList

def mcmcSearch(n, evalFunc, neighborFunc, initFunc, steps=10000):
    currentLoc = initFunc(n)
    currentVal = evalFunc(currentLoc)

    bestLoc = currentLoc
    bestVal = currentVal

    values = []
    bestValues = []

    for stepNum in range(steps):
        if (stepNum % 1000) == 0:
            print stepNum / 10000

        candidateNeighbor = neighborFunc(currentLoc)
        result = stepMaybeEfficientExpGreg(evalFunc, currentVal, candidateNeighbor)
        if result[0]:
            currentLoc = candidateNeighbor
            currentVal = result[1]

            if currentVal > bestVal:
                bestLoc = currentLoc
                bestVal = currentVal

                print bestVal

        values.append(currentVal)
        bestValues.append(bestVal)

    p.plot(values)
    p.plot(bestValues)


    p.show()
#    p.plot(bestValues)

#    p.show()

    return bestLoc, bestVal

def logDetCirc(l):
    return np.linalg.slogdet(circulant(l))[1]

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

def decToBin(num, binLength):
    if binLength == 0:
        return []

    else:
        lastBit = num % 2
        return decToBin(num >> 1, binLength - 1) + [lastBit]

def inflateList(x, newLength):
    return [x[int(i/newLength*len(x))] for i in range(newLength)]

def getMaxCircMatrix(x):
    bestDet = -float("Inf")
    bestMats = []
    bestLists = []
    allLists = allListsOfSizeX(x)

    for l in allListsOfSizeX(x):
#        print l

        mat = circulant(l)
        det = int(abs(np.linalg.det(mat))+0.001)
#        print "det", det
#        print "bestDet", bestDet

        if det > bestDet:
            bestDet = det
            bestMats = [mat]
            bestLists = [l]
#            print "better"

        elif det == bestDet:
            bestMats.append(mat)
            bestLists.append(l)
#            print "tied"

    return bestMats, bestLists, bestDet

def rotate(l, n):
    return l[-n:] + l[:-n]

def leadingZeros(l):
    counter = 0
    for x in l:
        if x == 0:
            counter += 1

        else:
            return counter

    return counter

def lexicalValue(l):
    counter = 0

    for i in range(len(l)):
        counter += l[i] * 2**i

    return -counter

def padIntegerWithZeros(x, maxLength):
    if x == 0:
        return "0"*maxLength

    assert log(x, 10) < maxLength

    return "0"*(maxLength-int(floor(log(x, 10)))-1) + str(x)

def strSum(l):
    returnString = ""

    for i in l:
        returnString += str(i)

    return returnString

def convertToInt(l):
    return int(strSum(l), 2)

def findBestRotation(l):
    lowestValue = float("Inf")
    bestList = None

#    print lexicalValue([0,1,1,0,1])

    for i in range(len(l)):
        candidate = rotate(l, i)

        value = convertToInt(candidate)

        if value < lowestValue:
            bestList = candidate
            lowestValue = value

    for i in range(len(l)):
        candidate = rotate(l, i)
        candidate.reverse()
        value = convertToInt(candidate)

        if value < lowestValue:
            bestList = candidate
            lowestValue = value

    return bestList

def plotEigs(firstRow):
    eigs = sorted([abs(i) for i in np.linalg.eig(circulant(firstRow))[0]])[::-1]

#    eigsNormalized = [i/eigs[0] for i in eigs]
    eigsNormalized = eigs

    n = len(firstRow)

    p.clf()

#    p.plot(np.linspace(0, 1, len(eigsNormalized)), eigsNormalized)
    p.plot(eigsNormalized)

    p.title(str(n))
    p.savefig("matrix_video_trash/eig" + padIntegerWithZeros(n,3) + ".png")

def getGreedyLists():
    listOfBests = []
    listOfBestLists = []

    for n in range(48, 48):
        print n

        bestList = randomGreedySearch(randomBits(n), getDeterminantOfCirc, \
            "max", 3*n)

        print bestList

#        pickle.dump(bestList, open("bestlist.p", "w"))

        bestVal = getDeterminantOfCirc(bestList)
        print bestVal
        if bestVal > 1:
            logBestVal = log(bestVal)
        else:
            logBestVal = 0

        listOfBests.append(logBestVal)
        listOfBestLists.append(findBestRotation([2*i for i in bestList]))

    return listOfBestLists

def getGroup(groups, k):
    for i, group in enumerate(groups):
        if k in group:
            return i

    return None


def addLinkToGroups(groups, k1, k2):
    k1GroupIndex = getGroup(groups, k1)
    k2GroupIndex = getGroup(groups, k2)

    if k1GroupIndex == None:
        if k2GroupIndex == None:
            return groups + [{k1: True, k2: True}]

        else:
            k2GroupCopy = groups[k2GroupIndex].copy()
            k2GroupCopy[k1] = True
            groupsCopy = groups[:]
            groupsCopy[k2GroupIndex] = k2GroupCopy

            return groupsCopy

    else:
        if k2GroupIndex == None:
            k1GroupCopy = groups[k1GroupIndex].copy()
            k1GroupCopy[k2] = True
            groupsCopy = groups[:]
            groupsCopy[k1GroupIndex] = k1GroupCopy

            return groupsCopy

        else:
            if k1GroupIndex == k2GroupIndex:
                return groups

            else:
                k1GroupCopy = groups[k1GroupIndex].copy()
                k2GroupCopy = groups[k2GroupIndex].copy()
                groupsCopy = groups[:]
                k1GroupCopy.update(k2GroupCopy)

                if k1GroupIndex < k2GroupIndex:
                    del groupsCopy[k2GroupIndex]
                    del groupsCopy[k1GroupIndex]

                else:
                    del groupsCopy[k1GroupIndex]
                    del groupsCopy[k2GroupIndex]

                groupsCopy.append(k1GroupCopy)

                return groupsCopy

if BRENT_LIST:
    BRENT_BINARY_STRING = "1,01,011,0111,01111,001011,0010111,00101111," + \
        "000101111,0000110111,00010110111,000110110111,0010111110111," + \
        "00001011101111,000100110101111,0000101101110111,00000101101110111," + \
        "000010011010101111,0000101011110011011,00000110110101110111"

    BRENT_DECIMAL_STRING = "45999,117623,340831,843119,638287,957175,1796839," + \
        "5469423,6774063,37463883,77446231,47828907,196303815,95151003," + \
        "1324935477,1822895095,430812063,2846677239,10313700815,6269629671," + \
        "26764629467,22992859983,92035379515,162368181483,226394696439," + \
        "631304341299,4626135339999"

    BRENT_DECIMAL_SIZES = range(21, 48)

    brentBinaryList = [[int(j) for j in list(i)] for i in \
        string.split(BRENT_BINARY_STRING, ",")]

    brentDecimalList = [decToBin(int(s), BRENT_DECIMAL_SIZES[i]) for i, s in \
        enumerate(string.split(BRENT_DECIMAL_STRING, ","))]

    greedyList = [i for i in getGreedyLists()]

    MAT_SIDE_LENGTH = 1000

    overallList = brentBinaryList + brentDecimalList + greedyList

    print [len(i) for i in overallList]


    listOfWinners = [inflateList(i, MAT_SIDE_LENGTH) for i in overallList]

    listOfMod4OverallLists = [[], [], [], []]

    for i, entry in enumerate(overallList):
        if i % 2 == 0:
            print i+1, int(floor((i+2)/2)), sum(entry),

        listOfMod4OverallLists[i % 4].append(entry)

    listOfWinners4 = [[inflateList(i, MAT_SIDE_LENGTH) for i in \
        listOfMod4OverallLists[j]] for j in range(4)]

    for i in range(4):
        arr = np.array(inflateList(listOfWinners4[i], MAT_SIDE_LENGTH))
        p.subplot(220 + i + 1)
        p.matshow(arr, fignum=False)

    p.show()

#    for i, firstRow in enumerate(brentBinaryList + brentDecimalList):
#        plotEigs(firstRow)

#    p.show()
#    p.clf()



#    print "low", listOfWinners

#    arr = np.array(inflateList(listOfWinners, MAT_SIDE_LENGTH))
#    p.matshow(arr)
#    p.show()


if EXHAUSTIVE:

    MAT_SIDE_LENGTH = 500
    listOfWinners = []

    MAX_NUM = 14

#    for n in range(1,MAX_NUM+1):

    for n in range(4, 5):

        bestMats, bestLists, bestDet = getMaxCircMatrix(n)
        for bestMat in bestMats:
            print bestMat
            eigs = np.linalg.eig(bestMat)

        print eigs
        print [np.abs(i) for i in eigs[0]]

        for bestList in bestLists:
            #print bestList
            pass

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
        print n

        bestList = randomGreedySearch(randomBits(n), getDeterminantOfCirc, \
            "max", 5*n)

        print bestList

        pickle.dump(bestList, open("bestlist.p", "w"))

        bestVal = getDeterminantOfCirc(bestList)
        print bestVal
        if bestVal > 1:
            logBestVal = log(bestVal)
        else:
            logBestVal = 0

        listOfBests.append(logBestVal)

if BASIC_TEST:
    m = np.dot(np.dot(dft(4), np.diag(np.array([3,-1,-1,-1]))), np.transpose(dft(4)))

if ARTIFICIAL_MLS:
    n = 7

#    fakeEigs = [(n+1)/2] + [sqrt(n+1)/2]*(n-1)

#    fakeEigs = [3.9999999999999973, 1.4142135623730954, 1.4142135623730954, 1.414213562373094, 1.414213562373094, 1.4142135623730945, 1.4142135623730945]

    fakeEigs = [(n+1)/2]

#    for i in range(int((n-1)/2)):
#        angle = np.exp(2*i/(n-1)*2*pi*1j)
#
#        fakeEigs.append(sqrt(n+1)/2*angle)
#        fakeEigs.append(sqrt(n+1)/2*angle)

    prescribedAngles = [85, 120.724, 172.152, 360-172.152, 360-120.724, 275]

    for i in range(n-1):
#        angle = np.exp((i+1)/n*2*pi*1j)

        angle = np.exp(prescribedAngles[i]/360*2*pi*1j)

        fakeEigs.append(sqrt(n+1)/2*angle)




    print fakeEigs


    fakeCirc = np.real(np.dot(np.dot(dft(n), np.diag(fakeEigs)), np.transpose(dft(n))))/n

    print np.linalg.eig(fakeCirc)[0]

#    print sum(fakeCirc[0])

    print fakeCirc

    p.matshow(fakeCirc)
    p.show()

if EXPLORE_LARGE_MLS:
    n = 31

    x = circulant([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1,
       1, 1, 0, 1, 1, 0, 0, 0])

    y = circulant(np.random.randint(0,2,n))

    print np.linalg.eig(x)[0]
    print np.linalg.eig(y)[0]

    print ""

    print sorted([np.angle(i, deg=True) for i in np.linalg.eig(x)[0]])
    print sorted([np.angle(i, deg=True) for i in np.linalg.eig(y)[0]])

    print ""

    x = sorted([abs(i) for i in np.linalg.eig(x)[0]])
    y = sorted([abs(i) for i in np.linalg.eig(y)[0]])

    print x
    print y

if MATRIX_EIG_ANALYSIS:
    n = 100

    A = circulant(np.random.randint(0,2,n))

    eigs = np.linalg.eig(A)[0]

    print np.linalg.slogdet(A)[1]

    p.plot([log(abs(i)) for i in eigs])

    p.show()

if MCMC_SEARCH:

    BRENT_BINARY_STRING = "1,01,011,0111,01111,001011,0010111,00101111," + \
        "000101111,0000110111,00010110111,000110110111,0010111110111," + \
        "00001011101111,000100110101111,0000101101110111,00000101101110111," + \
        "000010011010101111,0000101011110011011,00000110110101110111"

    BRENT_DECIMAL_STRING = "45999,117623,340831,843119,638287,957175,1796839," + \
        "5469423,6774063,37463883,77446231,47828907,196303815,95151003," + \
        "1324935477,1822895095,430812063,2846677239,10313700815,6269629671," + \
        "26764629467,22992859983,92035379515,162368181483,226394696439," + \
        "631304341299,4626135339999"

#    for n in range()

    n = 47

    evalFunc = logDetCirc
    neighborFunc = getRandomSwap
    initFunc = initFuncFloorNPlusOneOver2

    bestLoc, bestVal = mcmcSearch(n, evalFunc, neighborFunc, initFunc, steps=10000)

    s = 4626135339999

#    print evalFunc(decToBin(s, 47))

    eigs = np.linalg.eig(circulant(bestLoc))[0]

    print sorted([abs(i) for i in eigs])

    eigsOptimal = np.linalg.eig(circulant(decToBin(s, 47)))[0]

    print [abs(i) for i in eigsOptimal]

if EIG_ANALYSIS_2:

    n = 4

    s = "0111"

    print list(s)

    eigsOptimal = np.linalg.eig(circulant([int(i) for i in list(s)]))[0]

    print [abs(i) for i in eigsOptimal]


if EIG_ANALYSIS_SEARCH:

    numSolutions = 0

    n = 15

    def testIfGood(l):
        omega = 2*pi/n

        unityRoots = [omega*i for i in range(n)]

        cosines = [cos(i) for i in unityRoots]
        sines = [sin(i) for i in unityRoots]

        a = dot(cosines, l)
        b = dot(sines, l)

        return a**2 + b**2

    for l in allListsOfSizeX(n):
        if abs(testIfGood(l) - (n+1)/4) < 0.0001:
            numSolutions += 1

            print l

    print numSolutions

if EIG_EQUALITY_RESTRICTION:
#    if len(sys.argv) < 2:
#        n = 11
#    else:
#        n = int(sys.argv[1])

#    assert n % 4 == 3

    n = 1763

    MAX_N = 10000
    STOP_EARLY = True
    DONT_CHECK_SAME_GROUP_NUM = True

#    while n < MAX_N:
    for n in [15, 35, 143, 323, 899, 1763]:

        groups = []

        maximalPartitionEverFound = False
        maximalPartition = None

        groupNums = {}

        for k in range(n):
    #        for g in range(2, n):
            for g in [2, -2]:

    #            searchForGroups(k, g, s):


                groups = []

                for i in range(n):
                    groups = addLinkToGroups(groups, (i+k)%n, (i*g)%n)

                print k, g, len(groups)
#                for group in groups:
#                    print sorted(group.keys())

                if not (len(groups) in groupNums) or not DONT_CHECK_SAME_GROUP_NUM:

                    groupNums[len(groups)] = True

                    groups.sort(key=lambda g: len(g))

                    successTuple = findGoodGroup(groups[::-1], verbose=False, giveUpThreshold=27)

                    if not maximalPartitionEverFound:
                        if successTuple[0]:
                            maximalPartitionEverFound = True
                            maximalPartition = successTuple[1]
                            maximalAssignment = successTuple[2]

                            print maximalAssignment

                            for i, group in enumerate(groups):
                                print str(maximalAssignment[::-1][i]) + ":", sorted(group.keys())

                            if STOP_EARLY:
                                break



                else:
                    if DONT_CHECK_SAME_GROUP_NUM:
                        groupNums[len(groups)] = True

            if maximalPartitionEverFound and STOP_EARLY:
                break

        groupNums[len(groups)] = True

        if maximalPartitionEverFound:
            print "n =", n
            print "Maximal partition found!"
            print "Largest group count was", str(max(groupNums.keys())) + "."
            if isPrime(n):
                print "n is prime."

            else:
                if n in [15, 35, 63]:
                    print "n is already known to achieve the upper bound."

                else:
                    print "N IS COMPOSITE!!"
                    print "Decimal representation of maximal partition:"
                    print convertToInt(findBestRotation(maximalPartition))
                    bstr = strSum(findBestRotation(maximalPartition))
                    hstr = '%0*X' % ((len(bstr) + 3) // 4, int(bstr, 2))
                    print "Hexadecimal representation of maximal partition:"
                    print hstr
#                    print bstr

#            print maximalPartition

        else:
            print "No maximal partition found."
            print "Largest group count was", str(max(groupNums.keys())) + "."
            print "n =", n
            if isPrime(n):
                print "N IS PRIME??"

            else:
                if n in [15, 35, 63]:
                    print "N SHOULD ACHIEVE THE UPPER BOUND??"

                else:
                    print "n is composite."

        print ""
        print "--------"
        print ""

        n += 4


#    print groups


#    p.plot(range(1, MAX_NUM+1), listOfBests)
#    p.plot(range(1, MAX_NUM+1), [i/3*log(i) for i in range(1, MAX_NUM+1)])
#    p.plot(range(1, len(REAL_VALS)+1), [log(i) for i in REAL_VALS])

#    p.show()
