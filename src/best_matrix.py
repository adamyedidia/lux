from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as p
import matplotlib.patches as mpatches
from search import randomBits, randomGreedySearch
from math import log, sqrt, pi, floor, exp, cos, sin, ceil
import pickle
import string
from scipy.linalg import dft, circulant, hadamard, toeplitz, hankel
from scipy.signal import max_len_seq as mls
from scipy.integrate import quad
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
EIG_EQUALITY_RESTRICTION = False
URA_TEST = False
XOR_PRODUCT_TEST = False
TWIN_PRIME_INVESTIGATION = False
ALEX_TWIN_PRIME_IDEA = False
PRIME_TRIPLETS = False
IDENTITY_FUTZING = False
HADAMARD_DIAGONALIZABLE = False
PRIME_TRIPLETS_2 = False
LATIN_SQUARES = False
ICASSP_FIGS = False
ICASSP_TALK_TESTS = False
ICASSP_TALK_TESTS_2 = False
TOEPLITZ_TESTS = False
TOEPLITZ_TESTS_2 = False
TOEPLITZ_TESTS_3 = False
TOEPLITZ_TESTS_4 = False
DIFFERENT_DEPTHS_TEST = False
SZEGO_TEST = False
SZEGO_TEST_2 = False
ALT_PROJ = False
RANDOM_BLOCK_MAT = False
SF_BLOCK_MAT = False
COMPARE_SV = False
COMPARE_DEPTHS = False
RESIDUES_TEST = False
MSE_OF_WEIRD_SEQS = True

# compute a^b mod c
def modularExponentiation(a, b, c):
    if b == 1:
        return a

    if b % 2 == 0:
        return modularExponentiation((a*a) % c, b/2, c)

    if b % 2 == 1:
        return (a * modularExponentiation((a*a) % c, (b-1)/2, c)) % c

def quadResidue(a, p):
#    modExpResult = modularExponentiation(a, (p-1)/2, p)
    modExpResult = pow(a, int((p-1)/2), p)

    if modExpResult == 1:
        return True
    if modExpResult == -1 or modExpResult == p - 1:
        return False
    print modExpResult
    raise

def quartResidue(a, p):
    if quadResidue(a, p):
        modExpResult = pow(a, int((p-1)/4), p)

        if modExpResult == 1:
            return True
        elif modExpResult == -1 or modExpResult == p - 1:
            return False
        else:
            raise

    return False 

def octResidue(a, p):
    if quartResidue(a, p):
        modExpResult = pow(a, int((p-1)/8), p)

        if modExpResult == 1:
            return True
        elif modExpResult == -1 or modExpResult == p - 1:
            return False
        else:
            print a,modExpResult
            raise

    return False         

def gram(mat):
    return np.dot(np.transpose(mat), mat)

def otherGram(mat):
    return np.dot(mat, np.transpose(mat))

def alexQuadResidueOfProductOfTwoPrimes(a, p, q):
    quadResidueModP = quadResidue(a, p)
    quadResidueModQ = quadResidue(a, q)

    if (quadResidueModP and quadResidueModQ) or \
        ((not quadResidueModP) and (not quadResidueModQ)):

        return True
    return False

def circhank(x):
    return np.flip(circulant(x),0)

def product(l):
    returnValue = 1

    for i in l:
        returnValue *= i

    return returnValue

def ura(n):
    returnSequence = []

    for i in range(n):
        if i == 0:
            returnSequence.append(1)
        elif quadResidue(i, n):
            returnSequence.append(1)
        else:
            returnSequence.append(0)

    return returnSequence

def generalizedURA(n, identityValue, listOfPrimes, mode="oneZero"):

    returnSequence = []

    for i in range(n):
        returnSequence.append(generalAssignValueToZeroOrOne(i, n, identityValue, listOfPrimes, mode))

    return returnSequence

def logHBE(n):
    if n % 4 == 0:
        return n/2 * log(n)

    elif n % 2 == 0:
        return log(2) + log(n-1) + (n-2)/2 * log(n-2)

    else:
        return 1/2 * log(2*n-1) + (n-1)/2 * log(n-1)

def logU01(n):
    return logHBE(n+1) - n*log(2)

def accountForOrbit(x, r, n, accountedFor):
    accountedFor[x] = True
    currentNum = x

    while True:
        currentNum = (currentNum * r) % n

        if currentNum == x:
            break

        accountedFor[currentNum] = True

def binaryTriangular(n):
    return np.triu(np.ones((n,n)))

def bandDiagonal(n, k):
    tri = np.tri(n, k=k)

    return np.multiply(tri, np.transpose(tri))

def assignChoice(returnArray, x, r, n, val):
    if val == 0:
        currentNum = (-x) % n
    elif val == 1:
        currentNum = x % n
    else:
        raise

    returnArray[currentNum] = 1
    addedAlready = {currentNum: True}

#    print x, val

    while True:
        currentNum = (currentNum * r) % n

        if currentNum in addedAlready:
            break

        returnArray[currentNum] = 1
        addedAlready[currentNum] = True

def dissectChoices(p1, p2, r, accountedFor):
    n = p1*p2
    choices = []
    while len(accountedFor) < n:
        for i in range(n):
            if not i in accountedFor:
                choices.append(i)
                accountForOrbit(i, r, n, accountedFor)
                accountForOrbit((-i) % n, r, n, accountedFor)
                break

    print "choices", choices
    return choices

def incrementBinaryList(l, i=0):
    n = len(l)

    if l[i] == 0:
        l[i] = 1
    elif l[i] == 1:
        l[i] = 0
        if i < n-1:
            incrementBinaryList(l, i+1)

def assembleConstruction(p1, p2, r, choices, choiceAssignment):
    # multiples of p1 are 1's, multiples of p2 are 0's, 0 is a 0
    n = p1*p2
    returnArray = [0]*n

    for i in range(1,p2):
        returnArray[p1*i] = 1

    for i, choice in enumerate(choices):
        assignChoice(returnArray, choice, r, n, choiceAssignment[i])

    return returnArray

def average(l):
    return sum(l)/len(l)

def makeTwinPrimeConstruction(p1, p2):
    assert p1 + 2 == p2
    assert isPrime(p1)
    assert isPrime(p2)

    n = p1*p2

    accountedFor = {0: True}
    choices = {}

    if p1 % 4 == 3:
        r = 2
    elif p1 % 4 == 1:
        r = -2
    else:
        raise

    for i in range(1,p2):
        accountedFor[p1*i] = True

    for i in range(1, p1):
        accountedFor[p2*i] = True

    choices = dissectChoices(p1, p2, r, accountedFor)

    numChoices = len(choices)
    if numChoices > 10:
        print "I give up, there are", numChoices, "choices and that's just too many!"
        return -1, -1, -1

    satisfyingChoiceAssignments = []
    satisfyingArrays = []

    for l in allListsOfSizeX(numChoices):
        candidateArray = assembleConstruction(p1, p2, r, choices, l)

#        print "trying assgt", l

        if testIfMaximalEfficient(candidateArray):
            satisfyingChoiceAssignments.append(l)
            satisfyingArrays.append(candidateArray)

    return satisfyingArrays, choices, satisfyingChoiceAssignments

def compareLists(l1, l2):
    for i, j in zip(l1, l2):
        if i != j:
            return False
    return True

def generalAssignValueToZeroOrOne(i, n, identityValue, listOfPrimes, mode="oneZero"):

    if i == 0:
        return identityValue

    else:
        listOfPrimeFriends = []
        listOfPrimeEnemies = listOfPrimes[:]

        for prime in listOfPrimes:
            if i % prime == 0:
                listOfPrimeFriends.append(prime)
                listOfPrimeEnemies.remove(prime)

#        print i, listOfPrimeFriends, listOfPrimeEnemies

        legendreProduct = 1

        coreI = i
        for primeFriend in listOfPrimeFriends:
            coreI /= primeFriend

        for primeEnemy in listOfPrimeEnemies:

            legendreProduct *= -1

            if not quadResidue(int(coreI), primeEnemy):

                legendreProduct *= -1


        if len(listOfPrimeFriends) > 0:
            return 0
            # this is a HACK!

        if legendreProduct == -1:
            if mode == "oneZero":
                return 1
            elif mode == "oneNegativeOne":
                return -1

        else:
            if mode == "oneZero":
                return 0
            elif mode == "oneNegativeOne":
                return 1
            else:
                raise

def alexAssignValueToZeroOrOne(i, p1, p2):
    if i == 0:
        return 0
    elif i % p1 == 0:
        return 1
    elif i % p2 == 0:
        return 0
    elif alexQuadResidueOfProductOfTwoPrimes(i, p1, p2):
        return 1
    else:
        return 0

def makeAlexTwinPrimeConstruction(p1, p2):
    assert p1 + 2 == p2
    assert isPrime(p1)
    assert isPrime(p2)

    n = p1*p2

    returnList = []

    for i in range(n):
        returnList.append(alexAssignValueToZeroOrOne(i, p1, p2))

    return returnList

def makeTwinPrimeConstructionCheckBothRs(p1, p2):
    assert p1 + 2 == p2
    assert isPrime(p1)
    assert isPrime(p2)

    n = p1*p2

    satisfyingChoiceAssignments = []
    satisfyingArrays = []
    satisfyingRs = []

    for r in [2, -2]:

        accountedFor = {0: True}
        choices = {}

        for i in range(1,p2):
            accountedFor[p1*i] = True

        for i in range(1, p1):
            accountedFor[p2*i] = True

        choices = dissectChoices(p1, p2, r, accountedFor)

        numChoices = len(choices)

        for l in allListsOfSizeX(numChoices):
            candidateArray = assembleConstruction(p1, p2, r, choices, l)

            if testIfMaximalEfficient(candidateArray):
                satisfyingChoiceAssignments.append(l)
                satisfyingArrays.append(candidateArray)
                satisfyingRs.append(r)

    return satisfyingArrays, choices, satisfyingChoiceAssignments, satisfyingRs

def evaluate2DOccluder(arr):
    freqs = np.fft.fft2(arr)

    return np.sum(np.sum(np.log(np.abs(freqs)), 0), 0)

def logDetCircEfficient(l):
    f = np.fft.fft(l)

    returnSum = 0

    for eig in f:
        if abs(eig) <= 0:
            return -1e10

        returnSum += log(abs(eig))

    return returnSum

def miEfficient(l, gamma):
    f = np.fft.fft(l)
    n = len(l)

    returnSum = 0

    for eig in f:
#        if abs(eig) <= 0:
 #           return -1e10

        returnSum += log(gamma*abs(eig)**2/n**2 + 1)

    return returnSum



def testIfMaximalEfficient(l):
    logdet = logDetCircEfficient(l)
    logBound = logU01(len(l))
#    print logdet, logU01(len(l))

    smallDiff = 1e-10*min(logdet, logBound)

    # check if they're equal
    if abs(logdet - logU01(len(l))) < smallDiff:
        return True

    else:
        return False

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
            if testIfMaximalEfficient(firstRow):
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


def isFirstTwin(p1):
    if isPrime(p1) and isPrime(p1+2):
        return True
    return False

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

def allListsOfSizeXPlusMinusOne(x):
    if x == 0:
        return [[]]

    else:
        oneLess = allListsOfSizeXPlusMinusOne(x-1)
        return [i + [-1] for i in oneLess] + [i + [1] for i in oneLess]

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

    assert log(x+0.0001, 10) < maxLength

    return "0"*(maxLength-int(floor(log(x, 10)))-1) + str(x)

def strSum(l):
    returnString = ""

    for i in l:
        returnString += str(i)

    return returnString

def convertToInt(l):
    return int(strSum(l), 2)

def xorProduct(l1, l2):
    return [[(i + j) % 2 for j in l2] for i in l1]

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

def findAssignment(n):
    DONT_CHECK_SAME_GROUP_NUM = True
    STOP_EARLY = True
    maximalPartitionEverFound = False

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

                        for i, group in enumerate(groups):
                            print str(maximalAssignment[::-1][i]) + ":", sorted(group.keys())

                        return successTuple

            else:
                if DONT_CHECK_SAME_GROUP_NUM:
                    groupNums[len(groups)] = True

        if maximalPartitionEverFound and STOP_EARLY:
            break

    return False, False, False

def sieveOfEratosthenes(maxN):
    primalityDict = {}
    listOfPrimes = []

    for i in range(2, maxN+1):
        primalityDict[i] = True

    primeCounter = 2

    while primeCounter <= maxN/2 + 1:
        if primalityDict[primeCounter]:
            compositeCounter = 2*primeCounter

            while compositeCounter <= maxN:
                primalityDict[compositeCounter] = False

                compositeCounter += primeCounter

        primeCounter += 1

    for i in range(2, maxN+1):
        if primalityDict[i]:
            listOfPrimes.append(i)

    return primalityDict, listOfPrimes

def fixedBitIncrement(listOfIndices):
    newListOfIndices = listOfIndices[:]

#    print "start"

    counter = 0

    while counter < len(listOfIndices):
#        print newListOfIndices

        newListOfIndices[counter] += 1

        if counter == len(listOfIndices) - 1:
            break
        elif newListOfIndices[counter] == newListOfIndices[counter + 1]:
            newListOfIndices[counter] = counter
        else:
            break

        counter += 1

#    print newListOfIndices

    return newListOfIndices

def subSum(l, s):
    allLists = allListsOfSizeXPlusMinusOne(len(l))

    for listOfSizeX in allLists:
        if dot(listOfSizeX, l) == s:
            return (True, listOfSizeX)

    return (False, None)

def allDotProducts(a):
    returnList = []

    for i in range(len(a)):
        returnList.append(dot(a, rotate(a, i)))

    return returnList

def fuzzyLookup(array, index):
    floorIndex = int(floor(index))
    ceilIndex = int(ceil(index))

#    print floorIndex, ceilIndex, array


    residue = index % 1

    arrayBelow = array[floorIndex]
    arrayAbove = array[ceilIndex]


    return (1-residue) * arrayBelow + residue * arrayAbove

def getStretchedSubsetOfLength(occluder, n, leftEdgeMidPoint, \
    rightEdgeMidPoint):

    returnList = []

    for i in np.linspace(leftEdgeMidPoint, rightEdgeMidPoint-1e-8, n):
        returnList.append(fuzzyLookup(occluder, i))

    return returnList

def getToeplitzLikeTransferMatrixWithVariedDepth(occluder, d1, d2):
    assert abs(d1+d2-1) < 1e-8
    assert (len(occluder) % 2) == 1

    returnArray = []

    n = int((len(occluder)+1)/2)
    for i in range(n):
        sceneCoords = np.linspace(0, len(occluder)-1, n)

        leftEdgeMidPoint = sceneCoords[i]*d2 + 0*d1
        rightEdgeMidPoint = sceneCoords[i]*d2 + (len(occluder)-1)*d1

    #    print leftEdgeMidPoint, rightEdgeMidPoint

        snapshot = getStretchedSubsetOfLength(occluder, n, leftEdgeMidPoint, \
            rightEdgeMidPoint)

        returnArray.append(snapshot)

    return np.flip(np.transpose(np.array(returnArray)), 0)

def getToeplitzRectMatrixWithVariedDepthExplicitIndices(occluder, n1, n2):
    returnArray = []

    n = len(occluder)

    for i in range(n2):
        returnArray.append(occluder[i:i+n1])

    return np.flip(np.transpose(np.array(returnArray)), 0)    

def getToeplitzRectMatrixWithVariedDepth(occluder, d1, d2):
    returnArray = []

    n = len(occluder)

    n1 = int(len(occluder)*d1)
    n2 = n - n1 + 1

    for i in range(n2):
        returnArray.append(occluder[i:i+n1])

    return np.flip(np.transpose(np.array(returnArray)), 0)

def getRandomOccluderWithGivenBlockSize(n, blockSize):
    returnList = []

    numBlocks = int(n/blockSize)
    remainder = n % numBlocks

    for i in range(numBlocks):
        returnList.extend([1*(random.random() > 0.5)]*blockSize)

    returnList.extend([1*(random.random() > 0.5)]*remainder)

    return returnList

def getAverageMIRandomOccluderWithGivenBlockSize(n, blockSize, gamma, numSamples=100):
    returnVal = 0

    for _ in range(numSamples):
        occ = getRandomOccluderWithGivenBlockSize(n, blockSize)
#        print miEfficient(occ, gamma)

        returnVal += miEfficient(occ, gamma)


    returnVal /= numSamples

    return returnVal

def toeplitzFMaker(tFunc):
    def toeplitzF(lamda, n):
        returnSum = 0

        for k in range(-n+1, n):
            returnSum += tFunc(k, n)*np.exp(1j*k*lamda)

        return returnSum

    return toeplitzF

def szegoFMaker(bigF, tFunc, n):
    def toeplitzF(lamda):
        returnSum = 0

        for k in range(-n+1, n):
            returnSum += tFunc(k, n)*np.exp(1j*k*lamda)

        return bigF(returnSum)

    return toeplitzF

def szegoFMakerSV(bigF, tFunc, n):
    def toeplitzF(lamda):
        returnSum = 0

        for k in range(-n+1, n):
            returnSum += tFunc(k, n)*np.exp(1j*k*lamda)

        return bigF(np.abs(returnSum))

    return toeplitzF

def getEquivalentCirculant(k, tFunc, n):
    toeplitzF = toeplitzFMaker(tFunc)

    returnSum = 0

    for j in range(n):
        returnSum += toeplitzF(2*pi*j/n, n) * np.exp(2*pi*1j*j*k/n)/n

    return returnSum

def holeTFuncMaker(alpha):
    def tFunc(k, n):
        if abs(k) < alpha*n or k == alpha*n:
            return 1/n
        return 0
    return tFunc

def getToeplitzFromTFunc(tFunc, n):
    return toeplitz([tFunc(k, n) for k in range(0,-n,-1)], \
        [tFunc(k, n) for k in range(n)])

def getEigNorms(mat):
    return sorted(np.abs(np.linalg.eig(mat)[0]))[::-1]

def renormalizeFreqs(freqs):
    newFreqs = []
    newFreqs.append((n+1)/2)

    for freq in freqs[1:]:
        newFreqs.append(freq/abs(freq)*sqrt(n+1)/2)

    return np.array(newFreqs)

def roundToZeroOrOne(vals):
    newVals = []

    for val in vals:
        if abs(val - 1) < abs(val):
            newVals.append(1+0j)
        else:
            newVals.append(0+0j)

    return np.array(newVals)

def alternatingProjections(n):
    roundedVals = [1*(random.random()<0.5) for _ in range(n)]

    iterationCounter = 0
    maxIterations = 3

    while iterationCounter < maxIterations:
        freqs = np.fft.fft(roundedVals)
#        print freqs
        print np.abs(freqs)
        normalizedFreqs = renormalizeFreqs(freqs)
        print np.abs(normalizedFreqs)
#        print normalizedFreqs
        vals = np.fft.ifft(normalizedFreqs)
        print vals
        roundedVals = roundToZeroOrOne(vals)
        print roundedVals

        iterationCounter += 1

def admm(n):
    apologeticVals = [1*(random.random()<0.5) for _ in range(n)]

    iterationCounter = 0
    maxIterations = 1000

    angerFactor = 1

    rage = np.array([0+0j]*n)
    primalRage = np.array([0+0j]*n)

    while iterationCounter < maxIterations:
        freqs = np.fft.fft(apologeticVals)
#        print "a", np.abs(freqs)

        normalizedFreqs = renormalizeFreqs(freqs)

#        rage += normalizedFreqs - freqs

#        print rage

#        primalRage += np.fft.ifft(normalizedFreqs) - apologeticVals

#        print primalRage

#        angryFreqs = normalizedFreqs + rage

        vals = np.fft.ifft(normalizedFreqs)
#        print "b",vals
#        print "c",angryVals

#        print [abs(i) for i in np.fft.fft(vals)]

#        print apologeticVals, vals

        rage += angerFactor * (vals - apologeticVals)

#        print rage

        angryVals = vals + rage

        roundedVals = roundToZeroOrOne(vals)

        print sum([log(abs(i)) for i in np.fft.fft(roundedVals)])

#        print "d",[int(i) for i in np.real(roundedVals)]

        apologeticVals = roundedVals - rage

#        print "e",apologeticVals

    #    print roundedVals, vals

#        print "f",rage

        iterationCounter += 1

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

#    n = 3*7*11*19

    n = 231
    MAX_N = 233
    STOP_EARLY = True
    DONT_CHECK_SAME_GROUP_NUM = True

    while n < MAX_N:
#    for n in [15, 35, 143, 323, 899, 1763]:

        groups = []

        maximalPartitionEverFound = False
        maximalPartition = None


        maximalPartitionEverFound, maximalPartition, maximalAssignment = findAssignment(n)

        if maximalPartitionEverFound:
            print "n =", n
            print "Maximal partition found!"
    #        print "Largest group count was", str(max(groupNums.keys())) + "."
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
#            print "Largest group count was", str(max(groupNums.keys())) + "."
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

if URA_TEST:

    n = 195

#    print xorProduct([1,1,0],[1,0,1])

    seq = ura(n)
    twoDSeq = np.array(xorProduct(seq, seq))

    x = np.abs(np.fft.fft2(twoDSeq))

    x[0][0] = 50

    p.matshow(x)
    p.show()

#    p.matshow(np.array(xorProduct(seq, seq)), cmap='gray')
#    p.show()
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

    overallList = brentBinaryList + brentDecimalList

    while n < 5000:
#        if isPrime(n):
        if True:

            print n

            seq = ura(n)

#            print seq

            A = circulant(seq)
            eigs = np.linalg.eig(A)[0]

            print np.linalg.slogdet(A)[1]
            print np.linalg.slogdet(circulant(overallList[n-1]))[1]

            eigs = sorted([abs(i) for i in eigs])
            eigs.reverse()

#            print eigs

            p.plot([eigs[0]] + [i for i in eigs[1:]])
#            p.plot([eigs[0]/((n+1)/2)] + [i/(sqrt(n+1)/2) for i in eigs[1:]])
            p.show()

        n += 4

if XOR_PRODUCT_TEST:

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

    overallList = brentBinaryList + brentDecimalList

    n = 15

    print [abs(i) for i in np.fft.fft(overallList[n-1])]
    tapestry = np.abs(np.fft.fft2(xorProduct(overallList[n-1],overallList[n-1])))

    preTapestryRandom = np.random.randint(0,high=2,size=(n,n))
    tapestryRandom = np.abs(np.fft.fft2(preTapestryRandom))

    preSquareTapestryRandom = np.random.randint(0,high=2,size=n)
    squareTapestryRandom = np.abs(np.fft.fft2(xorProduct(preSquareTapestryRandom,
        preSquareTapestryRandom)))

    squareTapestryRandom[0][0] = 10

    p.matshow(squareTapestryRandom, cmap = "gray")
    p.show()

    print np.abs(np.fft.fft2(np.array(xorProduct(overallList[n-1],overallList[n-1]))))
    print np.abs(np.fft.fft2(np.random.randint(0,high=2,size=(n,n))))
    print np.abs(np.fft.fft2(np.array(xorProduct(preSquareTapestryRandom,
        preSquareTapestryRandom))))

    print evaluate2DOccluder(xorProduct(overallList[n-1],overallList[n-1]))
    print evaluate2DOccluder(np.random.randint(0,high=2,size=(n,n)))
    print evaluate2DOccluder(xorProduct(preSquareTapestryRandom,
        preSquareTapestryRandom))

if TWIN_PRIME_INVESTIGATION:
    p1 = 3

    while p1 < 10000:
        if isFirstTwin(p1):
            p2 = p1+2
            print "Twin prime found!"
            print p1, p2, p1*p2

#            satisfyingArrays, choices, satisfyingChoiceAssignments, satsifyingRs = \
#                makeTwinPrimeConstructionCheckBothRs(p1, p2)

            satisfyingArrays, choices, satisfyingChoiceAssignments = \
                makeTwinPrimeConstruction(p1, p2)

            if satisfyingArrays == -1:
                pass

            else:
                print "Choices were:"
                print choices
                print "Satisfying choice assignments were:"

                for satisfyingChoiceAssignment in satisfyingChoiceAssignments:
                    print satisfyingChoiceAssignment

#            print "Satisfying Rs were:"

#            for r in satsifyingRs:
#                print r

            print "--------------"

        p1 += 2

if ALEX_TWIN_PRIME_IDEA:
    p1 = 3

    while p1 < 10000:
        if isFirstTwin(p1):
            p2 = p1+2
            print "Twin prime found!"
            print p1, p2, p1*p2


            returnList = makeAlexTwinPrimeConstruction(p1, p2)

            if testIfMaximalEfficient(returnList):
                print "Nice! Alex was right!"
            else:
                print "Boo. Alex was wrong."

            print "--------------"

        p1 += 2

if PRIME_TRIPLETS:

    MAX_P = 10000

    primalityDict, listOfPrimes = sieveOfEratosthenes(MAX_P)

    numPrimes = 4
    listOfIndices = range(numPrimes)

    while listOfIndices[-1] < MAX_P:

#        print listOfIndices

        listOfIndices = fixedBitIncrement(listOfIndices)

        listOfPrimesInProduct = [listOfPrimes[i] for i in listOfIndices]

        allPrimesInProductAre3Mod4 = True

        for primeInProduct in listOfPrimesInProduct:
            if primeInProduct % 4 != 3:
                allPrimesInProductAre3Mod4 = False

#        if n % 4 == 3:
        if allPrimesInProductAre3Mod4:
            resultTuple = subSum([1] + listOfPrimesInProduct, 1)

            if resultTuple[0]:
                print "Success!"
                print listOfPrimesInProduct, resultTuple[1]

if IDENTITY_FUTZING:

#    print np.linalg.det(circulant([1,1,1,1,0]))

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

    overallList = brentBinaryList + brentDecimalList

    listOfPrimes = [3, 5, 7, 13]
    n = product(listOfPrimes)

#    for i in linspace()

    #for idVal in np.linspace(0.540,0.542,1000):

#    idVal = 0.5*(sqrt(5) - 1)
#    idVal = 0.5*(sqrt(37) - 5) + 0.0001
    idVal = 0
#    idVal = 0.5

#    g = generalizedURA(n, idVal, listOfPrimes, mode="oneZero")
    g = generalizedURA(n, idVal, listOfPrimes, mode="oneNegativeOne")

    brent = overallList[n-1]
    brent = [int((i-1/2)*2) for i in brent]

    print "first rows"
    print ""

    print"paley", g, sum(g)
    print "brent", brent, sum(brent)

    print "-----"

    print ""
    print "dots"
    print ""

    print "paley", allDotProducts(g)
    print "brent",allDotProducts(brent)

    print "-------"

    print ""
    print "eigs"
    print ""

    print "paley",[abs(i) for i in np.linalg.eig(circulant(g))[0]]
    print "brent",[abs(i) for i in np.linalg.eig(circulant(brent))[0]]


#    print circulant(g)


    print "-------"

    print ""
    print "det"
    print ""

    print "paley", np.linalg.det(circulant(g)), idVal
    print "brent", np.linalg.det(circulant(brent))

    print "--------"

    print ""
    print "logdet"
    print ""

    print "paley",np.linalg.slogdet(circulant(g))[1]
    print "brent",np.linalg.slogdet(circulant(brent))[1]

    #for i in np.linspace(0,1,100):

#    phi = 0.618033988
#    print np.linalg.det(circulant([1,phi,1,0,0]))

#    print [abs(i) for i in np.linalg.eig(circulant([1,1,1,1,0]))[0]]
#    print [abs(i) for i in np.linalg.eig(circulant([1,0.5,1,0,0]))[0]]
#    print sqrt(6)/2



#    for i in np.linspace(0,1,100):
#        print sum([log(abs(j)) for j in np.linalg.eig(circulant([1,i,1,0,0]))[0][1:]]), i

if HADAMARD_DIAGONALIZABLE:
    n = 64

    h = hadamard(n)

    d = [100*random.random() for _ in range(n)]
#    d = [1 for _ in range(n)]


    print d

    m = np.dot(np.dot(h, np.diag(d)), np.transpose(h))

    print m
    p.matshow(m)
    p.colorbar()
    p.show()

if PRIME_TRIPLETS_2:

    MAX_P = 10000

    primalityDict, listOfPrimes = sieveOfEratosthenes(MAX_P)

    numPrimes = 3
    listOfIndices = range(numPrimes)

    while listOfIndices[-1] < MAX_P:

#        print listOfIndices

        oldBiggest = listOfIndices[-1]

        listOfIndices = fixedBitIncrement(listOfIndices)

        if listOfIndices[-1] > oldBiggest:
            print listOfIndices[-1]


        def testForSuccess(p, q, r):
            if p*(r-q+1) + q - 2 == 0:
                print "Success!"
                print (p-1)*(q-1) + p-4 + (r-1)*p - (r-1)*(q-1)

                if (p-1)*(q-1) + p-4 + (r-1)*p - (r-1)*(q-1) == 0:

                    print "DOUBLE SUCCESS!"
                    print p, q, r #p*(r-q+1) + q

#        if n % 4 == 3:
        p = listOfPrimes[listOfIndices[0]]
        q = listOfPrimes[listOfIndices[1]]
        r = listOfPrimes[listOfIndices[2]]
        testForSuccess(p, q, r)
        p = listOfPrimes[listOfIndices[0]]
        q = listOfPrimes[listOfIndices[2]]
        r = listOfPrimes[listOfIndices[1]]
        testForSuccess(p, q, r)
        p = listOfPrimes[listOfIndices[1]]
        q = listOfPrimes[listOfIndices[0]]
        r = listOfPrimes[listOfIndices[2]]
        testForSuccess(p, q, r)
        p = listOfPrimes[listOfIndices[1]]
        q = listOfPrimes[listOfIndices[2]]
        r = listOfPrimes[listOfIndices[0]]
        testForSuccess(p, q, r)
        p = listOfPrimes[listOfIndices[2]]
        q = listOfPrimes[listOfIndices[0]]
        r = listOfPrimes[listOfIndices[1]]
        testForSuccess(p, q, r)
        p = listOfPrimes[listOfIndices[2]]
        q = listOfPrimes[listOfIndices[1]]
        r = listOfPrimes[listOfIndices[0]]
        testForSuccess(p, q, r)

if LATIN_SQUARES:
    a = 1
    b = 0
    c = 0
    d = 0
    e = 0
    f = 6

    cayleyTableMat = np.array([
        [e,a,b,c,f,d],
        [a,e,d,f,c,b],
        [b,f,e,d,a,c],
        [c,d,f,e,b,a],
        [d,c,a,b,e,f],
        [f,b,c,a,d,e]
    ])

    cayleyTableMat = np.array([
        [e,a,b,c,d,f],
        [a,e,d,f,b,c],
        [b,f,e,d,c,a],
        [c,d,f,e,a,b],
        [d,c,a,b,f,e],
        [f,b,c,a,e,d]
    ])

    cayleyTableMat = np.array([
        [a,b,c,d,e,f],
        [f,a,b,c,d,e],
        [e,f,a,b,c,d],
        [d,e,f,a,b,c],
        [c,d,e,f,a,b],
        [b,c,d,e,f,a]
    ])

    cayleyTableMat = np.array([
        [a,b,c,d,e],
        [d,e,a,b,c],
        [b,c,d,e,a],
        [e,a,b,c,d],
        [c,d,e,a,b]
    ])


    print np.transpose(np.linalg.eig(cayleyTableMat)[1])

    print np.linalg.det(dft(5)/sqrt(5))

    p.matshow(np.abs(np.linalg.eig(cayleyTableMat)[1]))
    p.colorbar()
    p.show()

    print np.linalg.eig(cayleyTableMat)


#####
if ICASSP_FIGS:
#    ax.grid(xdata=np.array(np.linspace(-0.5,11.5,12)))
#    ax.grid(ydata=np.array(np.linspace(-0.5,11.5,12)))

    first = False
    second = True

    if first:

        occ = [1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1]
        repeatedOcc = occ[:-1] + occ[:-1] + occ[:-1] + occ

        toep = toeplitz(occ[:11],occ[10:])
        hank = hankel([1/(100+(10-i)**2) for i in range(11)],[1/(100+i**2) for i in range(11)])


        weirdToep = getToeplitzLikeTransferMatrixWithVariedDepth(repeatedOcc, 0.25, 0.75)

        circ = np.transpose(circulant(occ))

    #    p.matshow(np.multiply(toep, hank), cmap="Greys_r")
        p.matshow(circ, cmap="Greys_r")
        p.matshow(weirdToep, cmap="Greys_r")

        p.show()

    if second:
        n = 43
#        p.matshow(np.array([ura(n), ura(n)]), cmap="Greys_r")
        randomArray = [1*(random.random()<0.5) for _ in range(n)]
        p.matshow(np.array([randomArray, randomArray]), cmap="Greys_r")
        p.show()


if ICASSP_TALK_TESTS:

    first = False

    if first:

        n = 1000
        occ = [1*(random.random() > 0.5) for _ in range(n)]

        spectrum = np.linalg.eig(circhank(occ))[0]

        histogram = [0]*200

        for eig in [abs(i)/sqrt(n) for i in sorted(spectrum)[:-1]]:
            print int(floor(eig*100))

            histogram[int(floor(eig*100))] += 1

        xTimesGauss = lambda x: exp(-x**2) * abs(x)

    #    p.plot(sorted([abs(i)/sqrt(n) for i in spectrum])[:-1])
    #    p.plot([i/100 for i in range(200)], [i/sqrt(n) for i in histogram])
    #    p.plot([i/100 for i in range(200)], [xTimesGauss(i/100) for i in range(200)])

    #    p.plot(sorted([exp()]))

    #    p.show()

    #lgN = 10
#    n = 103
    n = 1019
#    n = 10007
#    n = 100003

    flats = []
    randoms = []
    xs = []

    gamma = 1000

    for logN in range(3,17):

        print logN

        n = 2**logN - 1

        flats.append(miEfficient(mls(logN)[0], gamma))

        randomVals = [miEfficient([1*(random.random() > 0.5) \
            for _ in range(int(n))], gamma) for _ in range(10)]



        randoms.append(average(randomVals))
        xs.append(n)

    p.plot(xs, flats, "r-")
    p.plot(xs, randoms, "b-")

    ax = p.gca()

    redPatch = mpatches.Patch(color='red', label="Spectrally flat")
    bluePatch = mpatches.Patch(color='blue', label="Random")

    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')

    p.xlabel("$n$")
    p.ylabel("$I(x; y)$ (bits)")

    p.legend(handles=[redPatch, bluePatch])

    p.show()

if ICASSP_TALK_TESTS_2:
    n=10

    occ = [1*(random.random() > 0.5) for _ in range(n)]
    circ = circulant(occ)
    hank = circhank(occ)


    print occ

    p.matshow(circ, cmap="Greys_r")
    p.matshow(hank, cmap="Greys_r")
    p.show()

    print sorted([abs(i) for i in np.linalg.eig(circ)[0]])
    print sorted([abs(i) for i in np.linalg.eig(hank)[0]])

if TOEPLITZ_TESTS:

    MAX_N = 23

    for n in range(1, MAX_N, 2):
        bestLogDet = -float("Inf")
        bestL = None
        l = [0]*n

        while True:

            mat = toeplitz(l[:int((n+1)/2)][::-1], l[int((n-1)/2):])

            logDet = np.linalg.slogdet(mat)[1]

            if logDet > bestLogDet:
                bestLogDet = logDet
                bestL = l[:]
                bestMat = mat

            if sum(l) == n:
                break

            incrementBinaryList(l)

        bestMatIsCirculant = compareLists(bestL[:int((n-1)/2)], bestL[int((n+1)/2):])


        print n, int((n+1)/2), bestMatIsCirculant, np.linalg.det(bestMat), bestL

        p.matshow(bestMat, cmap="Greys_r")
        p.show()

        p.matshow(np.dot(bestMat, np.transpose(bestMat)), cmap="Greys_r")
        p.show()

        p.plot(sorted([abs(i) for i in np.linalg.eig(bestMat)[0]])[::-1])
        p.show()

if TOEPLITZ_TESTS_2:

    sims = []
    anas = []
    nums = []
    sim2s = []
#    ns = range(1, 1000)
    ns = range(6, 200, 4)

    alpha = 0.5

    gamma = 1

    for n in ns:

        if n % 100 == 0:
            print n

        k = int(n*alpha)
        toep = bandDiagonal(n, k)
#        toep = binaryTriangular(n)

#        p.matshow(toep)
#        p.show()

        sim = np.linalg.slogdet(gamma/n**2*np.dot(toep, np.transpose(toep)) + np.identity(n))[1]
        sim2eig = np.linalg.eig(toep)[0]

#        print sim2eig
#        print np.linalg.eig(np.dot(toep, np.transpose(toep)))[0]

        sim2 = sum([log(gamma*abs(i)**2/n**2 + 1) for i in sim2eig])
        ana = 4*pi*np.arcsinh(sqrt(gamma/n**2))*n/(2*pi)

        def integrand(lamda):
#            return log(2*gamma/n**2*(1-cos(lamda * n))/(1-cos(lamda)) + 1)
            return log(2*gamma/n**2*(1+2*cos(lamda*(1+k)/2)*sin(lamda*k/2)/sin(lamda/2))**2 + 1)

        num = quad(integrand, 0, 2*pi, limit=1000)[0]*n/(2*pi)

        sims.append(sim)
        sim2s.append(sim2)
        anas.append(ana)
        nums.append(num)

        print num, sim*2

    p.plot(ns, sims, "c-")
    p.plot(ns, sim2s, "r-")
    p.plot(ns, anas, "b-")
    p.plot(ns, nums, "g-")

    p.show()

if TOEPLITZ_TESTS_3:

    n = 100
#    k = 1j

    toep = np.transpose(binaryTriangular(n))/n
    print toep

#    print np.linalg.eig(toep)[1]

#    fourierVec = [np.exp(k*i) for i in range(n)]

    u, s, vh = np.linalg.svd(toep, full_matrices=True)

    p.matshow(u)
    p.colorbar()
    p.show()

    p.matshow(vh)
    p.colorbar()
    p.show()

    p.matshow(np.dot(u, vh))
    p.colorbar()
    p.show()

    print s

    print np.multiply(s,s)

#    print np.abs(fourierVec)
#    print np.abs(np.dot(toep, fourierVec))
#    print np.dot(toep, fourierVec)

#    print np.divide(np.dot(toep, np.array([np.exp(k*i) for i in range(n)])), np.array([np.exp(k*i) for i in range(n)]))

#    print sorted(np.abs(np.linalg.eig(toep)[0]))[::-1]
    print sorted(np.abs(np.linalg.eig(np.dot(toep, np.transpose(toep)))[0]))[::-1]

if TOEPLITZ_TESTS_4:
    alpha = 0.5
    n = 10

    tFunc = holeTFuncMaker(alpha)

#    circFirstRow = [getEquivalentCirculant(k, tFunc, n) for k in range(n)]

    toep = getToeplitzFromTFunc(tFunc, n)

    print toep

    print np.linalg.eig(toep)[0]

#    circ = circulant(circFirstRow)

#    print toep
#    print np.real(circ)

#    p.matshow(np.real(circ))
#    p.colorbar()
#    p.show()

#    print [(x,y)for x,y in zip(getEigNorms(toep), getEigNorms(circ))]

if DIFFERENT_DEPTHS_TEST:
    alpha = 0.5

    n = 1001

    d1 = 0.8
    d2 = 1 - d1

#    occluder = [0]*(int(n/4)) + [1]*(int(n/2+1)) + [0]*(int(n/4))
#    occluder = [0]*int((n-1)/2) + [1]*int((n+1)/2)
    occluder = [1*(random.random()<0.5) for _ in range(n)]


    mat = getToeplitzLikeTransferMatrixWithVariedDepth(occluder, d1, d2)

    p.matshow(mat)
    p.colorbar()
    p.show()

    p.matshow(gram(mat))
    p.colorbar()
    p.show()

#    p.matshow(otherGram(mat))
#    p.colorbar()
#    p.show()

if SZEGO_TEST:
    n = 20
    def bigF(x):
#        return log(np.abs(x)**2/n + 1)
        return x
#        return x**2


    def tFunc(k, n):
        return 1*(k>=0)

    def associatedToepF(lamda):
        return bigF(1/(1-np.exp(1j*lamda))/n)

    toep = getToeplitzFromTFunc(tFunc, n)

    toepEigs = np.linalg.eig(toep)[0]

    toepF = szegoFMaker(bigF, tFunc, n)

    a = sum([bigF(i) for i in toepEigs])/n

    print a

#    integ = quad(toepF, 0, 2*pi)
    integ = quad(associatedToepF, 0, 2*pi)

    b = integ[0]/2*pi
    print a, b

    print a/b

if SZEGO_TEST_2:
    n = 50000

    def bigF(x):
#        return log(x**2/n + 1)
        return x
#        return x**2


    def tFunc(k, n):
        return 1/n*(k>=0)

    toep = getToeplitzFromTFunc(tFunc, n)

    toepSVs = np.linalg.svd(toep)[1]

#    toepF = szegoFMakerSV(bigF, tFunc, n)

    a = sum([bigF(i) for i in toepSVs])/n

    print "a", a
#    integ = quad(toepF, 0, 2*pi)
#    print integ[1]

    b = integ[0]/2*pi

    print a/b

if ALT_PROJ:
    n = 25

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

    overallList = brentBinaryList + brentDecimalList


    bestPossible = sum([log(np.abs(i)) for i in np.fft.fft(overallList[n-1])])


    admm(n)
    print bestPossible

if RANDOM_BLOCK_MAT:
    gamma = 0.01

    logN = 13
    n = int(2**logN)

    averageMIs = []

    for logBlockSize in range(logN):

        blockSize = int(2**logBlockSize)

        print blockSize

        averageMI = getAverageMIRandomOccluderWithGivenBlockSize(n, blockSize, gamma, numSamples=100)

        averageMIs.append(averageMI)

    p.plot(range(logN), averageMIs)
    p.show()

if SF_BLOCK_MAT:
    gamma = 0.01

    logN = 15
    n = int(2**logN)

    mis = []
    xAxis = range(int(logN/2))
#    xAxis = range(int(logN-1))

    for logBlockSize in xAxis:

        blockSize = int(2**logBlockSize)

        logNumBlocks = logN - logBlockSize

        pattern = np.concatenate((mls(logNumBlocks)[0], np.array([0])), 0)
        print pattern, blockSize

        occ = np.repeat(pattern, blockSize)
        print occ

        mi = miEfficient(occ, gamma)

        mis.append(mi)

    p.plot(xAxis, mis)
    p.show()

if COMPARE_SV:

    d1 = 0.3
    d2 = 0.7

    n = 101

    occluder = [1*(random.random()>0.5) for _ in range(n)]

    mat = getToeplitzLikeTransferMatrixWithVariedDepth(occluder, d1, d2)
    rectMat = getToeplitzRectMatrixWithVariedDepth(occluder, d1, d2)

    u, s, vh = np.linalg.svd(mat)
    u2, s2, vh2 = np.linalg.svd(rectMat)

    print s

    p.plot(s)
    p.show()

    print s2

    print np.linalg.det(otherGram(mat))
    print np.linalg.det(otherGram(rectMat))

    print np.linalg.det(gram(mat))
    print np.linalg.det(gram(rectMat))

    p.plot(s2)
    p.show()

    p.matshow(mat)
    p.show()
    p.matshow(rectMat)
    p.show()

if COMPARE_DEPTHS:
    n = 101
    littleN = int((n+1)/2)
    occluder = [1*(random.random()>0.5) for _ in range(n)] 

    mis = []
    rectMis = []
    ds = []
    snr = 1

#    ds = np.linspace(0.01, 0.99, 99)
    ns = range(1, n)


    for n1 in ns:
        n2 = n - n1

        d1 = n1/n
        d2 = n2/n

        mat = getToeplitzLikeTransferMatrixWithVariedDepth(occluder, d1, d2)
        rectMat = getToeplitzRectMatrixWithVariedDepthExplicitIndices(occluder, n1, n2)

        rectN = np.shape(otherGram(rectMat))[0]

        ds.append(d1)


#        p.matshow(gram(snr*gram(rectMat) + np.identity(rectN)))
#        print np.linalg.det(gram(snr*gram(rectMat) + np.identity(rectN)))
#        p.colorbar()
#        p.show()

#        p.matshow(rectMat)
#        p.colorbar()
#        p.show()

#        print rectMat
#        print otherGram(rectMat)

#        p.matshow(otherGram(rectMat))
#        p.colorbar()
#        p.show()

        mis.append(np.linalg.slogdet(snr*otherGram(mat) + np.identity(littleN))[1])
        rectMis.append(np.linalg.slogdet(snr*otherGram(rectMat) + np.identity(rectN))[1])


    p.plot(ds, mis)
    p.plot(ds, rectMis)
    p.show()

if RESIDUES_TEST:
    p = 41

    print [quadResidue(a, p) for a in range(1, p)]
    print [pow(a, 2, p) for a in range(1, p)]

    print [quartResidue(a, p) for a in range(1, p)]
    print [pow(a, 4, p) for a in range(1, p)]

    print [octResidue(a, p) for a in range(1, p)]    
    print [pow(a, 8, p) for a in range(1, p)]

if MSE_OF_WEIRD_SEQS:
    