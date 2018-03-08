from __future__ import division
import numpy as np
from scipy.linalg import circulant as circ
from scipy.signal import max_len_seq as mls
from scipy.linalg import hankel
from math import sqrt, exp, pi, log
import matplotlib.pyplot as pl
import scipy.integrate as integrate
import sys
from numpy.random import chisquare
import random
from scipy.linalg import dft
from scipy.integrate import quad


generatePPlots = False
test = False
test2 = False
compareChiSquare = False
testFixedVariance = False
plotMILargeN = True

def circhank(x):
    return np.flip(circ(x),0)

def getCenteredVector(p, n):
    return np.random.binomial(1, p, n)/sqrt(p*(1-p)) - np.ones(n)*(p/sqrt(p*(1-p)))


def getCenteredVectorOld(p, n):
    returnList = []

    for _ in range(n):
        r = random.random()
        if r < p:
            returnList.append(sqrt(1-p)/sqrt(p))
        else:
            returnList.append(sqrt(p)/sqrt(1-p))

    return np.array(returnList)

def EWIntegrandMaker(i, w, p):
    def EWIntegrand(x):
#        eigPart = (p*(1-p)*x*x)/(2*w*i)
        eigPart = (p*(1-p)*x*x)/(2*w)
        expPart = abs(x)*exp(-x/2)/2

        return log(eigPart + 1, 2)*expPart
#        return x*expPart

    return EWIntegrand

def EW(i, w, p):
    return quad(EWIntegrandMaker(i, w, p), 0, np.inf)[0]

def oldEWIntegrandMaker(J, W, p):
    def oldEWIntegrand(x):
#        eigPart = (p*(1-p)*x*x)/(2*w*i)
        eigPart = abs(x)*exp(-x**2)
        expPart = (p*(1-p))*x**2

        alpha = 1 / (W + p*J)


        return log(alpha*expPart+1)*eigPart
#        return x*expPart

    return oldEWIntegrand

def oldEW(J, W, p):
    return 2*quad(oldEWIntegrandMaker(J, W, p), 0, np.inf)[0]

def newEWIntegrandMaker(J, W, p):
    def newEWIntegrand(x):
#        eigPart = (p*(1-p)*x*x)/(2*w*i)
        eigPart = x**2
        expPart = (p*(1-p))*abs(x)*exp(-x**2)

        alpha = 1 / (W + p*J)


        return alpha*eigPart*expPart
#        return x*expPart

    return newEWIntegrand

def newEW(J, W, p):
    alpha = 1 / (W + p*J)

    print "hi", alpha*p*(1-p)
    print "ho", 2*quad(newEWIntegrandMaker(J, W, p), 0, np.inf)[0]


    return 2*quad(newEWIntegrandMaker(J, W, p), 0, np.inf)[0] + log(alpha*p**2 + 1)


def chisquareAverage(numSamples=100000):
    runningSum = 0

    for _ in range(numSamples):
        runningSum += chisquare(2)

    return runningSum/numSamples

def empiricalEW(i, w, p, numSamples=1000):
    runningSum = 0

    for _ in range(numSamples):

        x = chisquare(2)

        eigPart = (p*(1-p)*x*x)/(2*w)

        runningSum += log(eigPart + 1, 2)

    return runningSum/numSamples

def empiricalEWFromMat(n, w, p, numSamples=500):
    runningSum = 0

    mat = getRandomCircMat(n, p)
    e = np.linalg.eig(mat)[0]


#    print [np.absolute(i) for i in e]

    for _ in range(numSamples):

        eigIndex = random.randint(1, n-1)

        x = np.absolute(e[eigIndex])

        eigPart = (x*x)/(w)

        runningSum += log(eigPart + 1, 2)

    return runningSum/numSamples

def MISumWithRealEigenvalues(mat, n, w, p):
    e = np.linalg.eig(mat)[0]

    runningSum = 0

    firstEigValue = log(e[0]**2/w + 1, 2)/n

    print firstEigValue

    runningSum += firstEigValue

    for i in range(1, n):
        runningSum += log(np.absolute(e[i])**2/w + 1, 2)/n
        if i % 100 == 0:
            print runningSum

    return runningSum

def MISum(n, w, p):
    runningSum = 0

    firstEigValue = log((p*sqrt(n))**2/w + 1, 2)/n
#    print firstEigValue

    runningSum += firstEigValue

    for i in range(2, int(n/2)):
        runningSum += 2*EW(i,w,p)/n
#        if i % 50 == 0:
#            print runningSum

    return runningSum

def newMISum(n, w, p):
    runningSum = 0

    firstEigValue = p*sqrt(n)**2/w
#    print firstEigValue

    runningSum += firstEigValue

    for i in range(2, int(n/2)):
        runningSum += 2*EW(i,w,p)
#        if i % 50 == 0:
#            print runningSum

    return runningSum

def getRandomCircMat(n, p):
    s = np.random.binomial(1,p,n)
    c = circ(s)

    normalizedC = c/sqrt(n)

    return normalizedC

def getMI(mat, n, w, p):
    combinedMat = np.dot(mat, np.transpose(mat))/w + np.identity(n)

    return np.linalg.slogdet(combinedMat)[1]/(n*log(2))

def getAverageRandomMI(n, w, p, numSamples=10):
    runningSum = 0

    for _ in range(numSamples):
        mat = getRandomCircMat(n, p)
        runningSum += getMI(mat, n, w, p)

    return runningSum / numSamples


def mean(l):
    return sum(l) / len(l)

def variance(l):
    m = mean(l)
    return mean([(i - m)**2 for i in l])

def Fw(w):
    alpha = sqrt(4*p*(1-p))/2
    return abs(w/alpha) * 1/alpha * exp(-(w/alpha)**2)

def estimateVarianceOfTotal(Q, numSamples=1000):
    samples = []

    for _ in range(numSamples):
        x = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        samples.append(sum(x))

    return np.var(np.array(samples))

def getOldMI(J, W, n, p):
    firstRowA = np.random.binomial(1,p, size=n)
    A = circ(firstRowA)

    alpha = 1/(W+p*J)



    firstMat = np.dot(A, np.transpose(A))


    overallMat = alpha*firstMat/n + np.identity(n)
#    overallMat = n/W*1/(W+p*J)*firstMat/n**2 + np.identity(n)

    return np.linalg.slogdet(overallMat)[1]

def getNewMI(J, W, n, p):

    firstRowA = np.random.binomial(1,p, size=n)
    A = circ(firstRowA)

    alpha = 1/(W+p*J)

    firstMat = np.dot(A, np.transpose(A))

    overallMat = alpha*firstMat/n**2 + np.identity(n)
#    overallMat = n/W*1/(W+p*J)*J/n*firstMat/n**2 + np.identity(n)

    return np.linalg.slogdet(overallMat)[1]

def getNewMISpectrallyFlat(J, W, n, p):

#    assert (floor(log(n, 2) + 1) == log(n, 2) + 1):

    firstRowA = mls(int(log(n, 2) + 1))[0]

    print firstRowA.astype(float)

    A = circ(firstRowA.astype(float))

    eigs = [abs(i) for i in np.linalg.eig(A)[0]]


    alpha = 1/(W+p*J)

    firstMat = np.dot(A, np.transpose(A))

    overallMat = alpha*firstMat/n**2 + np.identity(n)

#    pl.matshow(overallMat)
#    pl.show()
#    overallMat = n/W*1/(W+p*J)*J/n*firstMat/n**2 + np.identity(n)


    return np.linalg.slogdet(overallMat)[1]


def getSimNewMI(J, W, n):
    # worry about J later

    return newMISum(W, n, 0.5)

#def mutualInfoValue(p, numSamples):
#    for

if test:
    i = 2
    w = 0.01 # 20 dB SNR
    p = 0.5

    print empiricalEW(i, w, p)
    print EW(i, w, p)
    print chisquareAverage()

if test2:
    n = 1000
    i = 2
    w = 0.01 # 20 dB SNR
    p = 0.2

    mat = getRandomCircMat(n, p)

#    print MISumWithRealEigenvalues(mat, n, w, p)
#    print MISum(n, w, p)

    print EW(i, w, p)
    print empiricalEW(i, w, p)
    print empiricalEWFromMat(n, w, p)

if generatePPlots:
    n = 200
    w = 0.01 # 20 dB SNR

    ps = []
    misSim = []
    misAna = []

    for p in np.linspace(0, 1, 10):
        ps.append(p)
        misSim.append(getAverageRandomMI(n, w, p))
        misAna.append(MISum(n, w, p))

    print ps
    print misSim
    print misAna

    pl.plot(ps, misSim, "b-")
    pl.plot(ps, misAna, "r-")
    pl.show()

    sys.exit()

if compareChiSquare:

    n = 1000
    p = 0.5

#    s = (np.random.binomial(1,p,n)*2 - np.ones(n))
    s = getCenteredVector(p, n)

    print mean(s)
    print variance(s)



#    print s
    c = circ(s)
    normalizedC = c/sqrt(n)


    print normalizedC

    dftMat = dft(n)/sqrt(n)
    diagC = np.dot(np.dot(np.conj(dftMat), normalizedC), dftMat)

    print diagC

    diagDiagC = np.diagonal(diagC)

    pl.plot(np.multiply(np.conjugate(diagDiagC), diagDiagC))
    pl.show()

    pl.matshow(np.real(diagC))
    pl.show()

#    sys.exit()
#    h = circhank(s)
#    normalizedH = h/sqrt(n)

    e1 = np.linalg.eig(normalizedC)
#    e2 = np.linalg.eig(normalizedH)

#    print e1[0]
#    print e2[0]


#    x1 = sorted(np.multiply(e1[0], np.conjugate(e1[0])))
#    x2 = sorted(np.multiply(e2[0], np.conjugate(e2[0])))
    x1 = np.multiply(e1[0], np.conjugate(e1[0]))
#    x2 = np.multiply(e2[0], np.conjugate(e2[0]))
#    print x1

    x3 = chisquare(2, int((n-1)/2))

#    print x3.shape

    x3 = sorted(np.repeat(x3, 2))

#    print x3

    pl.plot(x1[:-1], "b-")
#    pl.plot(x2[:-1], "r-")
#    pl.plot(x3, "g-")
    pl.show()

    sys.exit()

if testFixedVariance:

    variances = []

    interestingNs = range(2, 30)

    for n in interestingNs:
#        scaling = 1/log(n)
#        scaling = 100
        scaling = 1

        print n

        D = np.diag(np.array([1/(i+1) for i in range(n)]))

        Q = scaling *1/n**2 *np.dot(np.dot(dft(n), D), np.transpose(dft(n)))
#        Q = np.identity(n)
#        Q = D

        variances.append(estimateVarianceOfTotal(Q))

    pl.plot(interestingNs, variances)
    pl.show()

if plotMILargeN:
    newMIs = []
    newMISFs = []
    oldMIs = []
    simNewMIs = []
    simOldMIs = []
    newnewMISFs = []

    J = 1
    W = 0.01
    p = 0.8

    alpha = 1/(W + p*J)

#    xAxis = range(200, 500, 25)
    xAxis = [2**i - 1 for i in range(2,10)]

#    numSamples = 10

    for n in xAxis:
        newMI = sum([getNewMI(J, W, n, p) for _ in range(10)])/10
        newMISF = getNewMISpectrallyFlat(J, W, n, p)
#        newMI = getNewMI(J, W, n, p)
#        oldMI = getOldMI(J, W, n, p)
        simNewMI = newEW(J, W, p)

        simpleFormula = log(p**2/4*alpha + 1, 2) + p*(1-p)/4*alpha


#        simOldMI = oldEW(J, W, p)



#        print simNewMI, simOldMI

    #    print n, oldMI, newMI

        newnewMISFs.append(simpleFormula)

        newMIs.append(newMI)
        newMISFs.append(newMISF)
#        oldMIs.append(oldMI/n)
        simNewMIs.append(simNewMI)
#        simOldMIs.append(simOldMI)


    pl.plot(xAxis, newMIs, "r-")
    pl.plot(xAxis, newMISFs, "g-")
    pl.plot(xAxis, newnewMISFs, "b-")
#    pl.plot(xAxis, oldMIs, "b-")
    pl.plot(xAxis, simNewMIs, color='r', marker='.')
#    pl.plot(xAxis, simOldMIs, color='b', marker='.')
    pl.show()

if False:

    n = 100
    s = mls(10)

    print np.linalg.det(circ(s))


if False:

    n = 300
    p = 0.5
    w = 0.2

    sums = []

    for p in np.linspace(0.5, 1.0, 100):

        s = np.random.binomial(1,p,n)
        c = circhank(s)

        normalizedC = c/sqrt(n)
    #    normalizedC = c/n

        e = np.linalg.eig(normalizedC)[0]
        sumLogEigs = 0
        for eig in e:
            sumLogEigs += log(eig*eig/w + 1, 2)

        sums.append(sumLogEigs)

    pl.plot(np.linspace(0.5, 1.0, 100), sums)
    pl.show()

    sys.exit()

    #print e

    x, bins = np.histogram(e[1:], bins=10, range=[-1, 1])

    #print x
    #print bins

    pl.plot([i/100 for i in range(-200, 200)], [Fw(w/100) for w in range(-200, 200)])
    pl.hist(e, bins=50, normed=1, range=[-1,1])
    pl.show()

    print integrate.quad(Fw, float("-Inf"), float("Inf"))


    #print s
    #print c
    #print e
