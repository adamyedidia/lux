from __future__ import division
import numpy as np
from scipy.linalg import circulant as circ
from scipy.signal import max_len_seq as mls
from scipy.linalg import hankel
from math import sqrt, exp, pi, log, floor, cos
import matplotlib.pyplot as pl
import scipy.integrate as integrate
import sys
from numpy.random import chisquare
import random
from scipy.linalg import dft
from scipy.integrate import quad
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


generatePPlots = False
generateOldPPlots = False
test = False
test2 = False
compareChiSquare = False
compareMIUniform = False
generateComparisonPlot = False
generatePStarComparisonPlot = False
comparePJ = False
uptTest = False
covTest = False
expTest = False
uniformTest = False
bestApertureSize = True
singleApertureSizeTest = False

def circhank(x):
    return np.flip(circ(x),0)

def getCenteredVector(p, n):
    return np.random.binomial(1, p, n)/sqrt(p*(1-p)) - np.ones(n)*(p/sqrt(p*(1-p)))

def alphaFunc(theta, J, W, p):
    return theta / (W + p*J)

def newEW(theta, J, W, p):
    alpha = alphaFunc(theta, J, W, p)

    return p*(1-p)*alpha + log(alpha*p**2 + 1, 2)

def EWExpIntegrandMaker(n, k, beta, theta, j, w, p):

    def EWExpIntegrand(x):
#        eigPart = (p*(1-p)*x*x)/(2*w*i)
        alpha = alphaFunc(theta, j, w, p)

        eigPart = 1/(2*n)*alpha*p*(1-p)*x*beta**(k/((n-1)/2))

        expPart = exp(-x/2)/2

#        print log(eigPart + 1, 2), expPart

        return log(eigPart + 1, 2)*expPart

#        return 1*expPart


#        return x*expPart

    return EWExpIntegrand



def EWGaussIntegrandMaker(n, theta, j, w, p):
    def EWGaussIntegrand(g):
        alpha = alphaFunc(theta, j, w, p)

        eigPart = 1/n*alpha*(sqrt(p*(1-p))*g + p*sqrt(n))**2

        expPart = 1/sqrt(2*pi) * exp(-g**2/2)

        return log(eigPart + 1, 2)*expPart
#        print log(eigPart + 1, 2), expPart

#        return 1*expPart

    return EWGaussIntegrand

def EWExp(n, k, beta, theta, j, w, p):
    result = quad(EWExpIntegrandMaker(n, k, beta, theta, j, w, p), 0, np.inf)[0]

#    print result

    return result

def EWGauss(n, theta, j, w, p):
    result = quad(EWGaussIntegrandMaker(n, theta, j, w, p), -np.inf, np.inf)[0]
#    print result

    return result

def MISumExp(n, beta, theta, j, w, p):
    runningSum = 0

    firstEigValue = EWGauss(n, theta, j, w, p)
#    print firstEigValue

    eigs = []

    runningSum += firstEigValue
#    eigs.append(firstEigValue)

    for k in range(1, int((n-1)/2)+1):
        eig = EWExp(n, k, beta, theta, j, w, p)

        runningSum += 2*eig

#        eigs.append(eig)
#        eigs.append(eig)

#        if i % 50 == 0:
#            print runningSum

#    print eigs

#    pl.plot(eigs)

    return runningSum

def getQExp(beta, theta, n):
    assert n % 2 == 1

    diagHalfArray = [beta**(i/((n-1)/2)) for i in range(1, int((n-1)/2) + 1)]

    diagArray = [1] + diagHalfArray + diagHalfArray

    assert len(diagArray) == n

    DStar = np.diag(diagArray)

#    pl.matshow(DStar)
#    pl.show()

    normalizedDFT = dft(n)/sqrt(n)

#    print DStar

#    pl.matshow(DStar)
#    pl.show()

#    pl.matshow(abs(np.dot(np.dot(np.transpose(np.conj(normalizedDFT)), DStar),
#        normalizedDFT)))
#    pl.show()

    return np.dot(np.dot(np.transpose(np.conj(normalizedDFT)), DStar),
        normalizedDFT)

def getPStar(theta, J, W):
    pStar = None
    bestMI = -1

    for p in np.linspace(0, 1, 1001):
        mi = newEW(theta, J, W, p)
        if mi > bestMI:
            pStar = p
            bestMI = mi

    return pStar

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

def getValueOfMIForIID(n, w, p, j):
    alpha = 1/(w+p*j)
    return log(p*p*alpha + 1, 2) + p*(1-p)*alpha

def spectrallyFlatLogTerm(k, w, j):
    return log(1/4/(w+j/2) * 1/(k**2) + 1)
#    return log(1/4/(w+j/2) * 1 + 1, 2)


def getSpectrallyFlat(n, w, j):
    runningSum = 0

    for k in range(2, int(n/2)):
        runningSum += 2*spectrallyFlatLogTerm(k, w, j)/n

    return runningSum

def getChristosCovMat(n, scaling):
    assert n % 2 == 1

    halfMainDiag = [1*scaling/(i+2) for i in range(int((n-1)/2))]

    firstEntry = [1/n]

    mainDiag = firstEntry + halfMainDiag + halfMainDiag[::-1]

    offDiag = [1]*(n-1)

    return np.diag(mainDiag) + np.flip(np.diag(offDiag, k=-1), 1)


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

def getMI(mat, S, jw, p):
    combinedMat = S * np.dot(mat, np.transpose(mat))/(n*(1/jw + p)) + np.identity(n)

    return np.linalg.slogdet(combinedMat)[1]/(log(2))

def getNewMI(A, S, jw, p):

    firstRowA = np.random.binomial(1,p, size=n)
    A = circ(firstRowA)

    alpha = S/(1/jw+p)

    firstMat = np.dot(A, np.transpose(A))

    overallMat = alpha*firstMat/n**2 + np.identity(n)
#    overallMat = n/W*1/(W+p*J)*J/n*firstMat/n**2 + np.identity(n)

    return np.linalg.slogdet(overallMat)[1]

def getAverageRandomMI(n, theta, j, w, p, numSamples=10):
    runningSum = 0

    for _ in range(numSamples):
        firstRowA = np.random.binomial(1,p, size=n)
        A = circ(firstRowA)

        runningSum += getNewMI(A, S, jw, p)

    return runningSum / numSamples

def apertureMIAnalytic(A, theta, J, w, p, alpha):

    gamma = alphaFunc(theta, J, w, p)

    n = A.shape[0]

    runningSum = 0

    runningSum += log(gamma * 1/n**2 * sum(A[0])**2 + 1, 2)

    eigs = []
    eigs.append(sum(A[0]))

    for j in range(1, n):

        runningSum += log(gamma * 2 * ((1-cos(j*2*pi*alpha))/(1-cos(2*pi*j/n))) * 1/n**2 + 1, 2)

#        print 2 * ((1-cos(j*2*pi*alpha))/(1-cos(2*pi*j/n)))

#        print (1-cos(j*2*pi*alpha))


        eigs.append(sqrt(2 * ((1-cos(j*2*pi*alpha))/(1-cos(2*pi*j/n)))))

    return runningSum, eigs

def getNewMIExp(A, Q, beta, theta, j, w, p):
    alpha = alphaFunc(theta, j, w, p)


    firstMat = np.dot(np.dot(A, Q), np.transpose(A))

    overallMat = alpha*firstMat/n**2 + np.identity(n)

#    eigs = sorted([abs(i) for i in np.linalg.eig(overallMat)[0]])[:-1]

#    print log(eigs[0], 2)

#    print sum([log(abs(i), 2) for i in eigs])

#    print log(np.linalg.eig(overallMat)[0][0], 2)

#    overallMat = n/W*1/(W+p*J)*J/n*firstMat/n**2 + np.identity(n)

    return np.linalg.slogdet(overallMat)[1]/log(2)

def getAverageRandomMIExp(n, beta, theta, j, w, p, numSamples=10):
    runningSum = 0

    Q = getQExp(beta, theta, n)

    for i in range(numSamples):
        firstRowA = np.random.binomial(1, p, size=n)
        A = circ(firstRowA)

        if i == 0:
            # DEBUG
#            pl.plot([abs(i) for i in np.linalg.eig(A)[0]])

#            print [abs(i) for i in np.linalg.eig(A)[0]]

#            pl.show()

            pass

        runningSum += getNewMIExp(A, Q, beta, theta, j, w, p)

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

def miBernoulliPStar(w, j):
    pStar = getPStar(w, j)
    p = pStar

    return -2*exp((pStar*j + w)/(p*(1-p)))*expIntegralEi(-(pStar*j + w)/(p*(1-p)))/log(4)

#def getAdamCovMat()

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

if generatePStarComparisonPlot:

    W = 1e7

    for color, theta in [("r", 1e7), ("orange", 10**7.5), ("y", 1e8), ("g", 10**8.5), ("c", 1e9),
        ('b', 10**9.5), ('m', 1e10)]:#, ("b", 100.0)]:

        snrs = []
        pStars = []

    #    for w in np.linspace(1, 10, 100):
        for jwInDB in np.linspace(-20, 20, 100):

            snrs.append(jwInDB)
    #        ws.append(w)

            J = 10**(jwInDB/10)*W

            pStar = getPStar(theta, J, W)

            pStars.append(pStar)

        pl.plot(snrs, pStars, color=color, linestyle="-")

#        for i, snr in enumerate(snrs):
#            if i % 5 == 0:
#                pl.plot(snr, pStars[i], color=color, marker="*",
#                    markersize=7)

    ax = pl.gca()

#    dottedLine = mlines.Line2D([], [], color='black', linestyle=':',
#                          label='Spectrally flat occluder')
#    thickLine = mlines.Line2D([], [], color='black', linestyle='-',
#                          label='Random on-off occluder, p = 0.5')
#    starredLine = mlines.Line2D([], [], color='black', marker='*',
#                            markersize=7, label='Random on-off occluder, p = p*')

    redPatch = mpatches.Patch(color='red', label=r"$\theta/W$ = 0 dB")
    orangePatch = mpatches.Patch(color='orange', label=r"$\theta/W$ = 5 dB")
    yellowPatch = mpatches.Patch(color='yellow', label=r"$\theta/W$ = 10 dB")
    greenPatch = mpatches.Patch(color='green', label=r"$\theta/W$ = 15 dB")
    cyanPatch = mpatches.Patch(color='cyan', label=r"$\theta/W$ = 20 dB")
    bluePatch = mpatches.Patch(color='blue', label=r"$\theta/W$ = 25 dB")
    magentaPatch = mpatches.Patch(color='magenta', label=r"$\theta/W$ = 30 dB")

#    bluePatch = mpatches.Patch(color='blue', label='1/J = -20 dB')

    pl.legend(handles=[redPatch, orangePatch, yellowPatch, greenPatch,\
        cyanPatch, bluePatch, magentaPatch])#, bluePatch])


    ax.grid()
#    ax.set_yscale("log")
    ax.set_ylabel("p*")
    ax.set_xlabel("J/W in dB")

    ax.set_ylim([0,1.05])
    ax.set_yticks(np.linspace(0,1,11))

    pl.show()

if generateComparisonPlot:

    W = 0.1


    for color, theta in [("orange", 0.01), ("r", 0.1), ("m", 1.0), ("b", 10.0), ("g", 100.0)]:#, ("b", 100.0)]:

        snrs = []
        misFlat = []
        misPStar = []

    #    for w in np.linspace(1, 10, 100):
        for jInDB in np.linspace(-30, 30, 100):

            snrs.append(jInDB)
    #        ws.append(w)

            J = 10**(jInDB/10)

            misFlat.append(newEW(theta, J, W, 0.5))
            pStar = getPStar(S, jw)

            print jw, S, pStar

            misPStar.append(newEW(S, jw, pStar))

        pl.plot(snrs, misFlat, color=color, linestyle=":")
        pl.plot(snrs, misPStar, color=color, linestyle="-")

        for i, snr in enumerate(snrs):
            if i % 5 == 0:
                pl.plot(snr, misPStar[i], color=color, marker="*",
                    markersize=7)

    ax = pl.gca()

    dottedLine = mlines.Line2D([], [], color='black', linestyle=':',
                          label='Spectrally flat occluder')
#    thickLine = mlines.Line2D([], [], color='black', linestyle='-',
#                          label='Random on-off occluder, p = 0.5')
    starredLine = mlines.Line2D([], [], color='black', marker='*',
                            markersize=7, label='Random on-off occluder, p = p*')

    orangePatch = mpatches.Patch(color='orange', label='S = 1/5')
    redPatch = mpatches.Patch(color='red', label='S = 1/2')
    magentaPatch = mpatches.Patch(color='magenta', label='S = 1')
    bluePatch = mpatches.Patch(color='blue', label='S = 2')
    greenPatch = mpatches.Patch(color='green', label='S = 5')
#    bluePatch = mpatches.Patch(color='blue', label='1/J = -20 dB')

    pl.legend(handles=[dottedLine, starredLine, orangePatch, redPatch, magentaPatch ,\
        bluePatch, greenPatch])#, bluePatch])


    ax.grid()
    ax.set_yscale("log")
    ax.set_ylabel("Mutual information (bits)")
    ax.set_xlabel("J/W in dB")

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
    n = 251
#    jw = 100 # -10 dB

    j = 1
    w = 1
    theta = 1000

    for color, beta in [("r", 1e-4), ("orange", 1e-3), ("y", 1e-2), ("g", 1e-1), ("b", 1)]:# ("c", 0.0)]:

        ps = []
        misSim = []
        misNewSim = []
        misAna = []


        for p in np.linspace(0, 1, 21):

            ps.append(p)
#            misSim.append(getAverageRandomMI(n, S, jw, p))
#            misNewSim.append(getNewMI(n, S, jw, p))
            misAnaInstance = MISumExp(n, beta, theta, j, w, p)
            misSimInstance = getAverageRandomMIExp(n, beta, theta, j, w, p, numSamples=200)
            print misAnaInstance/misSimInstance

            misAna.append(misAnaInstance)
            misSim.append(misSimInstance)




#        misAna.append(MISum(n, w, p))

        print misSim
        print misAna
        print ps


        pl.plot(ps, misSim, linestyle="--", color=color)
        pl.plot(ps, misAna, linestyle="-", color=color)
#        pl.plot(ps, misNewSim, linestyle=":", color=color)

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

    redPatch = mpatches.Patch(color='red', label=r"$\beta$ = -40 dB")
    orangePatch = mpatches.Patch(color='orange', label=r"$\beta$ = -30 dB")
    yellowPatch = mpatches.Patch(color='yellow', label=r"$\beta$ = -20 dB")
    greenPatch = mpatches.Patch(color='green', label=r"$\beta$ = -10 dB")
    bluePatch = mpatches.Patch(color='blue', label=r"$\beta$ = 0 dB")
#    bluePatch = mpatches.Patch(color='blue', label='1/J = -20 dB')

    pl.legend(handles=[dottedLine, thickLine, redPatch, orangePatch, yellowPatch ,\
        greenPatch, bluePatch])


    ax.grid()
#    ax.set_yscale("log")
    ax.set_ylabel("Mutual information (bits)")
    ax.set_xlabel("p")



    pl.show()

    sys.exit()

if covTest:
    n = 101
    scaling = 1

    cov = getChristosCovMat(n, scaling)

    print abs(np.dot(np.dot(dft(n), cov), np.transpose(dft(n))))/n

    pl.matshow(abs(np.dot(np.dot(dft(n), cov), np.transpose(dft(n)))))
    pl.show()



if uptTest:
    n = 80
#    upt = np.triu(np.ones((n,n)))
#    upt = np.identity(n)



#    upt = circ([0]*int(n/2) + [1]*int(n/2))
#    print sorted([abs(i) for i in np.linalg.eig(upt)[0]])
#    print np.linalg.slogdet(1/n**2 * np.dot(upt, np.transpose(upt)) + np.identity(n))


    oneTwoOneTopRow = [2, 1] + [0]*(n-3) + [1]
    oneTwoOneMat = circ(oneTwoOneTopRow)
    print sorted([abs(i) for i in np.linalg.eig(oneTwoOneMat)[0]])


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

if expTest:

    n = 1023
    beta = 0.99
    theta = 100
    j = 1
    w = 5
    p = 0.5

    a = np.array(mls(10)[0]).astype(float)

    A = circ(a)

    beta = 0.01

    Q = getQExp(beta, theta, n)

#    Q = np.identity(n)

#    beta = 0.99

    result = getNewMIExp(A, Q, beta, theta, j, w, p)

    print result

    alpha = alphaFunc(theta, j, w, p)

    print log(alpha/4 + 1, 2) + alpha*(1-beta)/log(2)/(4*log(1/beta))

if uniformTest:
    n = 1000
    beta = 1
    theta = 100
    j = 1
    w = 5
    p = 0.5

    a = np.random.uniform(0, 1, n)
    Q = np.identity(n)
    A = circ(a)
#    result = getNewMIExp(A, Q, beta, theta, j, w, p)

    print sum([getNewMIExp(A, Q, beta, theta, j, w, p) for _ in range(100)])/100

    alpha = alphaFunc(theta, j, w, p)

    print log(alpha/4 + 1, 2) + alpha/log(2)/12

#    print alpha/log(2)/4

#    print log(alpha/4 + 1, 2) + alpha/log(2)/4

#    print log(alpha/4 + 1, 2) + getNewMIExp

if bestApertureSize:

    n = 1000
    beta = 1
    theta = 10000
    j = 0
    w = 1

    MIs = []
    MIAnalytics = []

#    MI = getNewMIExp(A, Q, beta, theta, j, w, alpha)/log(2)
    Q = np.identity(n)


    baseMI = getNewMIExp(np.ones((n, n)), Q, beta, theta, j, w, 1)


    for alpha in np.linspace(0, 1, 101):

        numOnes = int(floor(alpha*n))
        numOnesFirst = int(numOnes/2)
        numOnesSecond = numOnes - numOnesFirst


        a = [1] * numOnesFirst + [0] * (n - numOnes) + [1] * numOnesSecond
        Q = np.identity(n)
        A = circ(a)

        print sum(A[0])

        MIAnalyticTuple = apertureMIAnalytic(A, beta, theta, j, w, alpha)

        MIAnalytic = MIAnalyticTuple[0]

        eigs = MIAnalyticTuple[1]

        gamma = alphaFunc(theta, j, w, alpha)
        MIAnalytic = sum([log(gamma * 1/n**2 * i**2 + 1, 2) for i in eigs])


#        pl.matshow(A)
#        pl.show()

        MI = getNewMIExp(A, Q, beta, theta, j, w, alpha)/log(2)

        MIs.append(MI)
        MIAnalytics.append(MIAnalytic)

    pl.plot(np.linspace(0, 1, 101), MIs)
    pl.plot(np.linspace(0, 1, 101), MIAnalytics)

    pl.show()

    pl.plot(np.linspace(0, 1, 101), [i/baseMI for i in MIs])
    pl.plot(np.linspace(0, 1, 101), [i/baseMI for i in MIAnalytics])

    pl.show()

if singleApertureSizeTest:
    n = 1000
    beta = 1
    theta = 10000
    j = 0
    w = 1
    alpha = 0.418932

    MIs = []
    MIAnalytics = []

    numOnes = int(floor(alpha*n))
    numOnesFirst = int(numOnes/2)
    numOnesSecond = numOnes - numOnesFirst


    a = [1] * numOnesFirst + [0] * (n - numOnes) + [1] * numOnesSecond
    Q = np.identity(n)
    A = circ(a)

    pl.matshow(A)
    pl.show()

    MIAnalyticTuple = apertureMIAnalytic(A, beta, theta, j, w, alpha)

    eigs = MIAnalyticTuple[1]

#        pl.matshow(A)
#        pl.show()

    #MI = getNewMIExp(A, Q, beta, theta, j, w, alpha)

    gamma = alphaFunc(theta, j, w, alpha)


    realEigs = [log(abs(i)**2 + 1, 2) for i in np.linalg.eig(A)[0]]

    pl.plot(sorted(realEigs)[::-1])
#    pl.plot(sorted(eigs))


    print sum([log(gamma * 1/n**2 * i**2 + 1) for i in realEigs])
#    print sum([log(gamma * 1/n**2 * i**2 + 1) for i in eigs])

    pl.show()


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
