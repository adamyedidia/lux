from __future__ import division
from math import pi, exp, sqrt
import random
import numpy as np


exploredPoints = {}
visitedPoints = {}

def fac(n):
    if n <= 0:
        return 1
    else:
        return n*fac(n-1)

def choose(n, r):
    return fac(n)/(fac(r)*(fac(n-r)))

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def normalize(distribution):
    s = sum(distribution)

    return [i/s for i in distribution]

def gaussianNoisePrior(x):
    return exp(-x**2/2)/sqrt(2*pi)

def getExpectedNearbyness(distribution):
    return sum([i*p for i, p in enumerate(distribution)])

def updateOnNeighbor(distribution, neighborResult, signalFunc):
    d = len(distribution) - 1

    oddsOfOneLess = 0
    oddsOfOneMore = 0

    newDistribution = []

    for i in range(d+1):
        oneMoreLikelihood = gaussianNoisePrior(signalFunc(i + 1) - neighborResult)
        oneLessLikelihood = gaussianNoisePrior(signalFunc(i - 1) - neighborResult)

        oneMorePrior = clamp(1 - i/d, 0, 1)
        oneLessPrior = clamp(i/d, 0, 1)

        oddsOfOneMoreGivenWeAreAtI = distribution[i] * oneMorePrior * oneMoreLikelihood
        oddsOfOneLessGivenWeAreAtI = distribution[i] * oneLessPrior * oneLessLikelihood

        oddsOfOneMore += oddsOfOneMoreGivenWeAreAtI
        oddsOfOneLess += oddsOfOneLessGivenWeAreAtI

    #    print getExpectedNearbyness(distribution), neighborResult
    #    print i, oneMorePrior * oneMoreLikelihood, oneLessPrior * oneLessLikelihood
    #    print i, oneMorePrior, oneMoreLikelihood, oneLessPrior, oneLessLikelihood
    #    print ""


    #    print oddsOfOneMoreUnnormalized, oddsOfOneLessUnnormalized

#        normalization = oddsOfOneMoreUnnormalized + oddsOfOneLessUnnormalized

#        if normalization > 0:

#        oddsOfOneMore = oddsOfOneMoreUnnormalized / normalization
#        oddsOfOneLess = oddsOfOneLessUnnormalized / normalization

        newDistribution.append(oddsOfOneMoreGivenWeAreAtI + oddsOfOneLessGivenWeAreAtI)

    normalization = oddsOfOneMore + oddsOfOneLess

    return normalize(newDistribution), oddsOfOneMore/normalization

def explore(point, signalFunc, noiseFunc):
    d = len(point)

    if tuple(point) in exploredPoints:
        return exploredPoints[tuple(point)]

    else:
        result = signalFunc(sum(point)) + noiseFunc()

        exploredPoints[tuple(point)] = result
        return result

def getNeighbor(point):
    d = len(point)

    flipIndex = random.randint(0, d-1)
    neighbor = point[:]
    neighbor[flipIndex] = 1 - neighbor[flipIndex]

    return neighbor

def tryOneStep(point, distribution, exploredThisPhase, signalFunc, noiseFunc):
    neighbor = getNeighbor(point)

    if tuple(neighbor) in visitedPoints or tuple(neighbor) in exploredThisPhase:
        return False, distribution, exploredThisPhase, None, None

#    print sum(point), getExpectedNearbyness(distribution), sum(neighbor)

    neighborResult = explore(neighbor, signalFunc, noiseFunc)

    exploredThisPhase[tuple(neighbor)] = True

    prevDistribution = distribution

    distribution, oddsOfOneMore = updateOnNeighbor(distribution, neighborResult, \
        signalFunc)

#    print "UPDATE --------"
#    print prevDistribution
#    print distribution
#    print "--------"


#    print oddsOfOneMore, getExpectedNearbyness(distribution), sum(point)

    if distribution[sum(point)] < 0.01:
        raise

    if oddsOfOneMore > 0.5:
        exploredThisPhase = {}
        return True, distribution, None, neighbor, oddsOfOneMore

    else:
        exploredThisPhase[tuple(point)] = True
        return False, distribution, exploredThisPhase, None, None

def modulateDistributionUp(distribution, probabilityOfSuccess):
    newUpDistribution = [0] + [probabilityOfSuccess * i for i in distribution[:-1]]

    newDownDistribution = [(1-probabilityOfSuccess) * i for i in distribution[1:]] + [0]

    newDistribution = [i+j for i,j in zip(newUpDistribution, newDownDistribution)]

    return normalize(newDistribution)

# Keep trying till it works!
def makeOneStep(point, distribution, signalFunc, noiseFunc):
    exploredThisPhase = {}

    while True:

        success, distribution, exploredThisPhase, neighbor, oddsOfOneMore = \
            tryOneStep(point, distribution, exploredThisPhase, signalFunc, noiseFunc)

#        print success, distribution, exploredThisPhase, neighbor, oddsOfOneMore

        if success:
            return neighbor, modulateDistributionUp(distribution, oddsOfOneMore)

def initializeUniform(d):
#    point = [1*(random.random() > 0.5) for _ in range(d)]

    point = [1]*7 + [0]*13
    distribution = normalize([choose(d, i) for i in range(d+1)])

    return point, distribution

def bayesWalk(d, signalFunc, noiseFunc):
    point, distribution = initializeUniform(d)

    while True:
        point, distribution = makeOneStep(point, distribution, signalFunc, noiseFunc)

        print sum(point), exploredPoints[tuple(point)], distribution[sum(point)]

#        print sum(point), point
#        print distribution
#        print distribution[sum(point)]
#        print ""

d = 20

signalFunc = lambda x: x
noiseFunc = lambda : np.random.normal()

bayesWalk(d, signalFunc, noiseFunc)
