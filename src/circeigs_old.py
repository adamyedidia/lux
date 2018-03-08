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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


generatePPlots = True
generateOldPPlots = False
test = False
test2 = False
compareChiSquare = False
compareMIUniform = False
generateComparisonPlot = False
comparePJ = False


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
        eigPart = (p*(1-p)*x)/(2*w)
        expPart = exp(-x/2)/2

        return log(eigPart + 1, 2)*expPart
#        return x*expPart

    return EWIntegrand

def EWIntegrandMakerPJ(k, w, p, j):
    def EWIntegrand(x):
        eigPart = 1/(w + p*j) * (p*(1-p)/(2*k**2)) * x*x
#        eigPart = 1/(w + p*j) * (p*(1-p)/(2)) * x

        expPart = exp(-x/2)/2

        return log(eigPart + 1, 2)*expPart

    return EWIntegrand

def EWPJ(k, w, p, j):
    result = quad(EWIntegrandMakerPJ(k, w, p, j), 0, np.inf)[0]
    return result

def MISumPJ(n, w, p, j):
    runningSum = 0

    firstEigValue = log((p*sqrt(n))**2/(w + p*j) + 1, 2)/n
#    print firstEigValue

    runningSum += firstEigValue

    for k in range(2, int(n/2)):
        runningSum += 2*EWPJ(k,w,p,j)/n
#        if i % 50 == 0:
#            print runningSum

    return runningSum

def spectrallyFlatLogTerm(k, w, j):
    return log(1/4/(w+j/2) * 1/(k**2) + 1)
#    return log(1/4/(w+j/2) * 1 + 1, 2)


def getSpectrallyFlat(n, w, j):
    runningSum = 0

    for k in range(2, int(n/2)):
        runningSum += 2*spectrallyFlatLogTerm(k, w, j)/n

    return runningSum

def miSpectrallyFlatOccluder(w, j):
    return log((1/4)*1/(w+j/2) + 1, 2)

def EWUniformIntegrandMaker(i, w):
    def EWIntegrand(x):
#        eigPart = (p*(1-p)*x*x)/(2*w*i)
        eigPart = (x)/(12*2*w)
        expPart = exp(-x/2)/2

        return log(eigPart + 1, 2)*expPart
#        return x*expPart

    return EWIntegrand

def EWUniform(i, w):
    return quad(EWUniformIntegrandMaker(i, w), 0, np.inf)[0]

def EW(i, w, p):
    return quad(EWIntegrandMaker(i, w, p), 0, np.inf)[0]

def chisquareAverage(numSamples=100000):
    runningSum = 0

    for _ in range(numSamples):
        runningSum += chisquare(2)

    return runningSum/numSamples

def empiricalEW(i, w, p, numSamples=100000):
    runningSum = 0

    for _ in range(numSamples):

        x = chisquare(2)

        eigPart = (p*(1-p)*x)/(2*w)

        runningSum += log(eigPart + 1, 2)

    return runningSum/numSamples

def empiricalChiSquare(i, w, p, numSamples=1000):
    runningSum = 0

    chis = []

    for _ in range(numSamples):

        x = np.random.normal()**2 + np.random.normal()**2

        runningSum += (p*(1-p)*x)/(2*w)
        chis.append(p*(1-p)*x/2)

    pl.plot(sorted(chis))

    return runningSum/numSamples

def empiricalEWFromManyMats(n, w, p, numSamples=100):
    runningSum = 0

    for _ in range(numSamples):
        runningSum += empiricalEWFromMat(n, w, p)

    return runningSum / numSamples

def empiricalEigFromMat(n, w, p, numSamples=None):
    runningSum = 0

    mat = getRandomCircMat(n, p)
    e = np.linalg.eig(mat)[0]

    pl.plot(sorted([np.absolute(i)**2 for i in e][1:]))

    pl.show()

    if numSamples == None:

        for eigIndex in range(1, n):
            x = np.absolute(e[eigIndex])

            eigPart = (x*x)/(w)

            runningSum += eigPart

        return runningSum/(n-1)

def empiricalEWFromMat(n, w, p, numSamples=None):
    runningSum = 0

    mat = getRandomCircMat(n, p)
    e = np.linalg.eig(mat)[0]


#    print [np.absolute(i) for i in e]
#    print "hi", np.absolute(log(e[0]**2/w + 1))/n



    if numSamples == None:

        for eigIndex in range(1, n):
            x = np.absolute(e[eigIndex])

            eigPart = (x*x)/(w)

            runningSum += log(eigPart + 1, 2)

        return runningSum/(n-1)

    else:
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

#    print firstEigValue

    runningSum += firstEigValue

    remainingEigs = e[1:]
    random.shuffle(remainingEigs)

    for i in range(1, n):
        runningSum += log(np.absolute(e[i])**2/w + 1, 2)/n
#        if i % 100 == 0:
#            print runningSum

    return runningSum

def MIUniformSum(n, w):
    runningSum = 0

    firstEigValue = log((0.5*sqrt(n))**2/w + 1, 2)/n
#    print firstEigValue

    runningSum += firstEigValue

    for i in range(2, int(n/2)):
        runningSum += 2*EWUniform(i,w)/n
#        if i % 50 == 0:
#            print runningSum

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

def getRandomCircMat(n, p):
    s = np.random.binomial(1,p,n)
    c = circ(s)

    normalizedC = c/sqrt(n)

#    pl.matshow(normalizedC)
#    pl.show()

    return normalizedC

def getRandomUniformCircMat(n):
    s = np.random.uniform(0,1,n)
    c = circ(s)

    normalizedC = c/sqrt(n)

    return normalizedC

def getMI(mat, n, w, p, j):
    combinedMat = np.dot(mat, np.transpose(mat))/(w + p*j) + np.identity(n)

    return np.linalg.slogdet(combinedMat)[1]/(n*log(2))

def getAverageRandomMI(n, w, p, j, numSamples=10):
    runningSum = 0

    for _ in range(numSamples):
        mat = getRandomCircMat(n, p)
        runningSum += getMI(mat, n, w, p, j)

    return runningSum / numSamples

def getAverageUniformRandomMI(n, w, numSamples=10):
    runningSum = 0

    for _ in range(numSamples):
        mat = getRandomUniformCircMat(n)
        runningSum += getMI(mat, n, w, 0.5)

    return runningSum / numSamples

def mean(l):
    return sum(l) / len(l)

def variance(l):
    m = mean(l)
    return mean([(i - m)**2 for i in l])

def Fw(w):
    alpha = sqrt(4*p*(1-p))/2
    return abs(w/alpha) * 1/alpha * exp(-(w/alpha)**2)

#def mutualInfoValue(p, numSamples):
#    for

#def miSpectrallyFlatOccluder(w, j):
#    return log((1/4)*1/(w+j/2) + 1, 2)

def expIntegral(t):
    return exp(-t)/t

def expIntegralEi(x):
    return -quad(expIntegral, -x, np.inf)[0]

def miBernoulliOneHalf(w, j):
    pStar = 0.5
    p = pStar

    return -2*exp((pStar*j + w)/(p*(1-p)))*expIntegralEi(-(pStar*j + w)/(p*(1-p)))/log(4)

def getPStar(w, j):
    return w/j*(sqrt(1+j/w) - 1)

def miBernoulliPStar(w, j):
    pStar = getPStar(w, j)
    p = pStar

    return -2*exp((pStar*j + w)/(p*(1-p)))*expIntegralEi(-(pStar*j + w)/(p*(1-p)))/log(4)


if test:
    i = 2
    w = 0.01 # 20 dB SNR
    p = 0.5

    print empiricalEW(i, w, p)
    print EW(i, w, p)
    print chisquareAverage()

if test2:
    n = 700
    i = 2
    w = 0.01 # 20 dB SNR
    p = 0.1

#    mat = getRandomCircMat(n, p)

#    print MISumWithRealEigenvalues(mat, n, w, p)
#    print getAverageRandomMI(n, w, p, 80)
#    print MISum(n, w, p)

#    sys.exit()

    print EW(i, w, p)
    print empiricalEW(i, w, p)
    print empiricalEWFromManyMats(n, w, p, numSamples=10)
#    print "#"
#    print empiricalChiSquare(i, w, p)
#    print empiricalEigFromManyMats(n, w, p)

if compareMIUniform:
    n = 2000
    w = 0.01 # 20 dB SNR

    ps = []
    misSim = []
    misAna = []

    misSim.append(getAverageUniformRandomMI(n, w, 80))
    misAna.append(MIUniformSum(n, w))

    print misSim, misAna

#    print ps
#    print misSim
#    print misAna

    sys.exit()

if generateComparisonPlot:


    for color, j in [("r", 0.01), ("m", 0.1), ("b", 1.0), ("g", 10.0)]:#, ("b", 100.0)]:

        snrs = []
        misFlat = []
        misBernoulli = []
        misPStar = []

    #    for w in np.linspace(1, 10, 100):
        for wInDB in np.linspace(-30, 10, 100):

            snrs.append(-wInDB)
    #        ws.append(w)

            w = 10**(wInDB/10)

            misFlat.append(miSpectrallyFlatOccluder(w, j))
            misBernoulli.append(miBernoulliOneHalf(w, j))
            misPStar.append(miBernoulliPStar(w, j))

        pl.plot(snrs, misFlat, color=color, linestyle=":")
        pl.plot(snrs, misBernoulli, color=color, linestyle="-")
        pl.plot(snrs, misPStar, color=color, linestyle="-")

        for i, snr in enumerate(snrs):
            if i % 5 == 0:
                pl.plot(snr, misPStar[i], color=color, marker="*",
                    markersize=7)

    ax = pl.gca()

    dottedLine = mlines.Line2D([], [], color='black', linestyle=':',
                          label='Spectrally flat occluder')
    thickLine = mlines.Line2D([], [], color='black', linestyle='-',
                          label='Random on-off occluder, p = 0.5')
    starredLine = mlines.Line2D([], [], color='black', marker='*',
                            markersize=7, label='Random on-off occluder, p = p*')

    redPatch = mpatches.Patch(color='red', label='1/J = 20 dB')
    orangePatch = mpatches.Patch(color='magenta', label='1/J = 10 dB')
    yellowPatch = mpatches.Patch(color='blue', label='1/J = 0 dB')
    greenPatch = mpatches.Patch(color='green', label='1/J = -10 dB')
#    bluePatch = mpatches.Patch(color='blue', label='1/J = -20 dB')

    pl.legend(handles=[dottedLine, thickLine, starredLine, redPatch, orangePatch, yellowPatch ,\
        greenPatch])#, bluePatch])


    ax.grid()
    ax.set_yscale("log")
    ax.set_ylabel("Mutual information per pixel (bits)")
    ax.set_xlabel("1/W in dB")

    pl.show()

if comparePJ:
    n = 10000
    j = 100
    w = 1
    pStar = getPStar(w, j)

    print "p=pstar", 2*MISumPJ(n, w, pStar, j)
    print "p=0.5", 2*MISumPJ(n, w, 0.5, j)
    print "spectrally flat", getSpectrallyFlat(n, w, j)
#    print miSpectrallyFlatOccluder(w, j)
#    print miBernoulliOneHalf(w, j)

if generateOldPPlots:

    n = 1000
    w = 1 # -10 dB

    j = 0

    ps = []
    misSim = []
    misAna = []


    for p in np.linspace(0, 1, 11):
        print p

        ps.append(p)
        misSim.append(getAverageRandomMI(n, w, p, j, 10))
        misAna.append(MISumPJ(n, w, p, j))
#        misAna.append(MISum(n, w, p))


    pl.plot(ps, misSim, linestyle="--", color="r")
    pl.plot(ps, misAna, linestyle="-", color="r")


    pl.show()

if generatePPlots:
    n = 250
    w = 0.01 # -10 dB

    j = 0

    for color, j in [("r", 0.01), ("orange", 0.1), ("y", 1.0), ("g", 10.0), ("c", 0.0)]:

        ps = []
        misSim = []
        misAna = []


        for p in np.linspace(0, 1, 11):
            print p

            ps.append(p)
            misSim.append(getAverageRandomMI(n, w, p, j, 1))
            misAna.append(MISumPJ(n, w, p, j))
#        misAna.append(MISum(n, w, p))

        print misSim
        print misAna
        print ps


        pl.plot(ps, misSim, linestyle="--", color=color)
        pl.plot(ps, misAna, linestyle="-", color=color)

#    print ps
#    print misSim
#    print misAna

    ax = pl.gca()

    dottedLine = mlines.Line2D([], [], color='black', linestyle='--',
                          label='Simulated mutual information')
    thickLine = mlines.Line2D([], [], color='black', linestyle='-',
                          label='Analytical mutual information')
#    starredLine = mlines.Line2D([], [], color='black', marker='*',
#                            markersize=10, label='Random on-off occluder, p = p*')

    redPatch = mpatches.Patch(color='red', label='1/J = 20 dB')
    orangePatch = mpatches.Patch(color='orange', label='1/J = 10 dB')
    yellowPatch = mpatches.Patch(color='yellow', label='1/J = 0 dB')
    greenPatch = mpatches.Patch(color='green', label='1/J = -10 dB')
#    bluePatch = mpatches.Patch(color='blue', label='1/J = -20 dB')

    pl.legend(handles=[dottedLine, thickLine, redPatch, orangePatch, yellowPatch ,\
        greenPatch])#, bluePatch])


    ax.grid()
#    ax.set_yscale("log")
    ax.set_ylabel("Mutual information per pixel (bits)")
    ax.set_xlabel("p")



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

if False:

    n = 100
    s = mls(10)

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
