from __future__ import division
import numpy as np
import random
from math import log, pi
import matplotlib.pyplot as p
from scipy.linalg import dft

l = 10
n = 20
m = 20
h = 100
w = 1e-10
cf = l*l/(pi*n*h)
P = 0.5
c = 10

plotMvsInfo = False
plotMvsInfoWithPrior = False
makeDarpaPlots = False
makeDarpaMetaPlots = True
makeDarpaInfoPlots = False
makeDarpaBasicPlots = False

SPECIFIC_RANDOM_OCCLUDER = [1.4835753031997756, 2.0298053160193885, \
    2.4069101950275806, 3.0273988090943806, 5.1278741978517655, \
        6.619829637105497, 7.146401156165197, 8.016002343396885]

OTHER_SPECIFIC_RANDOM_OCCLUDER = [0.1500259415264582, 0.41882400559435373, \
0.4930860464587916, 0.5665672113267117, 1.5265402667478678, 1.9276903753589414, \
2.366695509137492, 2.792729909984063, 2.8852517621675746, 3.928599240922458, \
4.39559560787726, 5.066230439912385, 5.814380478791086, 5.837854243111867, \
5.954388567784981, 6.359090100636612, 6.391318225106555, 6.49730666541989, \
6.57725999060928, 6.909778832416828, 6.977750088375601, 7.356182793301729, \
7.424378733485842, 7.693110407383116, 7.871815828412095, 8.058863832547216, \
8.186224411639945, 8.404658742399508, 8.951923879788884, 9.055868984271166]

OTHER_OTHER_SPECIFIC_RANDOM_OCCLUDER = [0.3113258306492894, 0.3511147814249216, 0.5120193704739495, 0.5678726653942179, 0.6249745650370753, 0.6330584299203912, 0.6773895011452191, 0.6835951869089618, 0.7514215882643316, 0.8168890544083007, 0.8492744093200688, 0.8936292393601686, 1.0223488737893693, 1.0898240786120195, 1.0963546569673988, 1.1510640778330972, 1.151586615698783, 1.288074277161243, 1.4832600944608432, 1.6135531674260406, 1.6552357258739903, 1.7513917116647715, 1.7850502950936453, 1.8561697526609433, 1.9974215394989625, 2.1119687998008274, 2.642270588702832, 2.793593307635016, 2.8351789702345287, 2.906711608429643, 3.0620266691207485, 3.140761761712172, 3.3264662851141127, 3.33111374751467, 3.4832754411642988, 3.504557799083429, 3.729640698373818, 3.7487784143074334, 3.837189263817619, 3.891387902047061, 3.9246462092684364, 4.045494460735991, 4.342575705181599, 4.363017041954188, 4.378605637741187, 4.453286788465418, 4.678050996922265, 4.70688638887453, 4.908493719265183, 4.969916193183201, 4.986567753797987, 5.116245050096069, 5.146138760494091, 5.284473391896042, 5.37413715663683, 5.4641995924089715, 5.497641408879035, 5.618453010623962, 5.654570036400738, 5.694747278144617, 5.8062363847666685, 5.816694151874802, 5.817574755811641, 5.863736334975567, 6.124454150839091, 6.202957410515246, 6.276854925112744, 6.423542579015119, 6.446693802644217, 6.773650621395713, 6.8527414984019845, 6.861796149877608, 7.0098982635344145, 7.026535045171776, 7.192090641886444, 7.1922342283879095, 7.380835827578461, 7.7138701369838785, 7.814116929372047, 7.878771989812049, 7.993746783735487, 8.068793852694466, 8.083458847798536, 8.330586015165743, 8.448421187563111, 8.451562872711163, 8.527991841512724, 8.588097245575897, 8.712712044771937, 8.763126624861941, 8.902946297296051, 9.00907873461793, 9.066960261138497, 9.073280961170369, 9.075490104727898, 9.312735836631221, 9.32521782121228, 9.398684044075878, 9.437648245410848, 9.71254462644471]

PINHOLE_OCCLUDER = [4.75, 5.25]
PINSPECK_OCCLUDER = [0, 4.75, 5.25]
EDGE_OCCLUDER = [5]

def randomOccluderDiscrete(n, probOn, maxX):
    transitionList = []

    for i in range(int(n)):
        if random.random() < probOn:
            transitionList.append(i/n*maxX)
            transitionList.append((i+1)/n*maxX)

    return transitionList

def getOneOverFSquaredPrior(n, c):
    dftMat = dft(n)

    oneOverFSquared = np.diag(np.array([c/i**2 for i in range(1,n+1)]))

    return np.dot(np.dot(dftMat, oneOverFSquared), dftMat)

def turnTransitionsSequenceIntoArray(queryOccluderAtPoint, arraySize, maxX):
    returnArray = []

    for i in np.linspace(0, maxX, arraySize):
        returnArray.append(queryOccluderAtPoint(i)*1)

    return np.array(returnArray)

def randomOccluder(numTransitions, maxX):
    transitionList = [random.random() * maxX for _ in range(numTransitions)]
    transitionList.sort()
    return transitionList

def getTotalOverRangeMaker(occluder):
    def getTotalOverRange(start, end, last):
        currentlyOn = False
        lastTransition = -float("Inf")
        totalSoFar = 0

        for transition in occluder + [last]:
            if currentlyOn:
                if transition > start:
                    if lastTransition < start:
                        if transition < end:
                            totalSoFar += transition - start
                        else:
                            totalSoFar += end - start
                    else:
                        if transition < end:
                            totalSoFar += transition - lastTransition
                        else:
                            totalSoFar += end - lastTransition

            lastTransition = transition

            if transition < end:
                currentlyOn = not currentlyOn
            else:
                return totalSoFar

        return totalSoFar

    def queryOccluderAtPoint(point):

        currentlyOn = False
        for transition in occluder:
            if transition > point:
                return currentlyOn

            else:
                currentlyOn = not currentlyOn

        return currentlyOn

    return getTotalOverRange, queryOccluderAtPoint

def getTransferMatrix(occ, m, n, l, h):
    getTotalOverRange = getTotalOverRangeMaker(occ)[0]

    matArray = []

    for i in range(n):
        matArray.append([])
        for j in range(m):
            start = i/n*l
            end = (i+1)/n*l

            center = (j+0.5)/m*l

            trueStart = (start+center)/2
            trueEnd = (center+end)/2

#            print trueStart, trueEnd

            contribution = getTotalOverRange(trueStart, trueEnd, l) * l/m / (pi*h)

            matArray[-1].append(contribution)

    return np.array(matArray).transpose()

def evaluateOccluder(occ, m, n, l, h, w, disp=False):
#    displayOccluder(occ)

    tm = getTransferMatrix(occ, m, n, l, h)

    if disp:
        print "info", np.linalg.slogdet(np.dot(tm.transpose(), tm)*m/w + np.identity(n))[1]/(log(2))

        p.matshow(tm)
        p.colorbar()
        p.show()

#    p.matshow(tm)
#    p.show()

#    if m % 10 == 0:
#        p.matshow(np.dot(tm.transpose(), tm)*m/w)
#        p.colorbar()
#        p.show()

    infoValue = np.linalg.slogdet(np.dot(tm.transpose(), tm)*m/w + np.identity(n))[1]/(log(2))

    return infoValue

def evaluateOccluderWithPrior(occ, m, n, l, h, w, c, disp=False):
#    displayOccluder(occ)

    tm = getTransferMatrix(occ, m, n, l, h)

    Q = getOneOverFSquaredPrior(n, c)

#    print Q

    if disp:
        print "info", np.linalg.slogdet(np.dot(np.dot(tm.transpose(), Q), tm)*m/w + np.identity(n))[1]/(log(2))

        p.matshow(tm)
        p.colorbar()
        p.show()

#    p.matshow(tm)
#    p.show()

#    if m % 10 == 0:
#        p.matshow(np.dot(tm.transpose(), tm)*m/w)
#        p.colorbar()
#        p.show()

    infoValue = np.linalg.slogdet(np.dot(tm.transpose(), tm)*m/w + np.identity(n))[1]/(log(2))

    return infoValue

def evaluateManyOccluders(occluderN, probOn, maxX, m, n, l, h, w, numSamples = 100):
    sumInfos = 0

    for _ in range(numSamples):
        sumInfos += evaluateOccluder(randomOccluderDiscrete(occluderN, probOn, maxX), \
            m, n, l, h, w)

    return sumInfos / numSamples

def evaluateManyOccludersWithPriors(occluderN, probOn, maxX, m, n, l, h, w, c, numSamples = 100):
    sumInfos = 0

    for _ in range(numSamples):
        sumInfos += evaluateOccluderWithPrior(randomOccluderDiscrete(occluderN, probOn, maxX), \
            m, n, l, h, w, c)

    return sumInfos / numSamples

def displayOccluder(occ, filename = None):

    queryOccluderPoint = getTotalOverRangeMaker(occ)[1]

    occArr = turnTransitionsSequenceIntoArray(queryOccluderPoint, 1000, l)

    occArr = [1-i for i in occArr]

    p.matshow(np.array([occArr]*100), cmap="Greys")

    if filename == None:
        p.show()
    else:
        p.savefig(filename)

#if makeDarpaInfoPlots:


if plotMvsInfoWithPrior:

    infoValuesPinhole = []
    infoValuesPinspeck = []
    infoValuesRandom = []
    infoValuesOtherRandom = []
    infoValuesLens = []
    infoValuesEdge = []
    upperBounds = []
    xAxis = []

    for m in range(3, 100):
        n = m
        cf = l*l/(pi*n*h)

#    for logW in np.linspace(-15, 0, 100):

#        w = 10**logW

        infoValuePinhole = evaluateOccluderWithPrior(PINHOLE_OCCLUDER, m, n, l, h, w, c)
        if m % 10 == 0 and m >= 40:
            infoValuePinspeck = evaluateOccluderWithPrior(PINSPECK_OCCLUDER, m, n, l, h, w, c, True)
        else:
            infoValuePinspeck = evaluateOccluderWithPrior(PINSPECK_OCCLUDER, m, n, l, h, w, c, False)
        infoValueRandom = evaluateOccluderWithPrior(SPECIFIC_RANDOM_OCCLUDER, m, n, l, h, w, c)
        infoValueOtherRandom = evaluateOccluderWithPrior(OTHER_OTHER_SPECIFIC_RANDOM_OCCLUDER, m, n, l, h, w, c)
        infoValueEdge = evaluateOccluderWithPrior(EDGE_OCCLUDER, m, n, l, h, w, c)
        infoValueLens = min(n, m) * log(1 + (m/w)*(l*l/(pi*h*n))**2, 2)

#        if m % 10 == 0:
#            p.matshow(getTransferMatrix(OTHER_OTHER_SPECIFIC_RANDOM_OCCLUDER, m,n,l,h))
#            p.colorbar()
#            p.show()


        earlyBound = (m/2)*log((1+n*cf*cf/(m*w))**2 + (n-1)*n**2*cf**4/(m**2*w**2), 2)
        lateBound = (n/2)*log((1+cf*cf/w)**2 + (n-1)*cf**4/w**2, 2)
        k = cf*cf/w

        hiSNRBound = n/2*log(n, 2) - (n-1) + n*log(k+1, 2)

#        print (m/w)*(l/(pi*h*n))**2

#        print earlyBound, lateBound, hiSNRBound, k

        infoValuesPinhole.append(infoValuePinhole)
        infoValuesPinspeck.append(infoValuePinspeck)
        infoValuesRandom.append(infoValueRandom)
        infoValuesOtherRandom.append(infoValueOtherRandom)
        infoValuesEdge.append(infoValueEdge)
        infoValuesLens.append(infoValueLens)

        upperBounds.append(min(min(earlyBound, lateBound), hiSNRBound))

        xAxis.append(m)
#        xAxis.append(10*log(k, 10))

        print 10*log(cf*cf/1e-10, 10)

    ax = p.gca()

    ax.set_xlabel("m")
#    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mutual Information (bits)")

#    p.plot(xAxis, infoValuesPinhole, "r-")
    p.plot(xAxis, infoValuesPinspeck, "g-")
    p.plot(xAxis, infoValuesRandom, "b-")
    p.plot(xAxis, infoValuesOtherRandom, "r-")
    p.plot(xAxis, infoValuesEdge, "c-")
    p.plot(xAxis, infoValuesLens, "m-")
    p.plot(xAxis, upperBounds, "k-")
    p.show()


if plotMvsInfo:

    infoValuesPinhole = []
    infoValuesPinspeck = []
    infoValuesRandom = []
    infoValuesOtherRandom = []
    infoValuesLens = []
    infoValuesEdge = []
    upperBounds = []
    xAxis = []

    for m in range(3, 100):
        n = m
        cf = l*l/(pi*n*h)

#    for logW in np.linspace(-15, 0, 100):

#        w = 10**logW

        infoValuePinhole = evaluateOccluder(PINHOLE_OCCLUDER, m, n, l, h, w)
        if m % 10 == 0 and m >= 40:
            infoValuePinspeck = evaluateOccluder(PINSPECK_OCCLUDER, m, n, l, h, w, True)
        else:
            infoValuePinspeck = evaluateOccluder(PINSPECK_OCCLUDER, m, n, l, h, w, False)
        infoValueRandom = evaluateOccluder(SPECIFIC_RANDOM_OCCLUDER, m, n, l, h, w)
        infoValueOtherRandom = evaluateOccluder(OTHER_OTHER_SPECIFIC_RANDOM_OCCLUDER, m, n, l, h, w)
        infoValueEdge = evaluateOccluder(EDGE_OCCLUDER, m, n, l, h, w)
        infoValueLens = min(n, m) * log(1 + (m/w)*(l*l/(pi*h*n))**2, 2)

#        if m % 10 == 0:
#            p.matshow(getTransferMatrix(OTHER_OTHER_SPECIFIC_RANDOM_OCCLUDER, m,n,l,h))
#            p.colorbar()
#            p.show()


        earlyBound = (m/2)*log((1+n*cf*cf/(m*w))**2 + (n-1)*n**2*cf**4/(m**2*w**2), 2)
        lateBound = (n/2)*log((1+cf*cf/w)**2 + (n-1)*cf**4/w**2, 2)
        k = cf*cf/w

        hiSNRBound = n/2*log(n, 2) - (n-1) + n*log(k+1, 2)

#        print (m/w)*(l/(pi*h*n))**2

#        print earlyBound, lateBound, hiSNRBound, k

        infoValuesPinhole.append(infoValuePinhole)
        infoValuesPinspeck.append(infoValuePinspeck)
        infoValuesRandom.append(infoValueRandom)
        infoValuesOtherRandom.append(infoValueOtherRandom)
        infoValuesEdge.append(infoValueEdge)
        infoValuesLens.append(infoValueLens)

        upperBounds.append(min(min(earlyBound, lateBound), hiSNRBound))

        xAxis.append(m)
#        xAxis.append(10*log(k, 10))

        print 10*log(cf*cf/1e-10, 10)

    ax = p.gca()

    ax.set_xlabel("m")
#    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mutual Information (bits)")

#    p.plot(xAxis, infoValuesPinhole, "r-")
    p.plot(xAxis, infoValuesPinspeck, "g-")
    p.plot(xAxis, infoValuesRandom, "b-")
    p.plot(xAxis, infoValuesOtherRandom, "r-")
    p.plot(xAxis, infoValuesEdge, "c-")
    p.plot(xAxis, infoValuesLens, "m-")
    p.plot(xAxis, upperBounds, "k-")
    p.show()

if makeDarpaPlots:
#    occ = randomOccluderDiscrete(4, 0.5, l)

#    getTotalOverRange, queryOccluderPoint = getTotalOverRangeMaker(occ)

#    displayOccluder(occ)

    infos = []
#    occluderScale = np.linspace(2, 100, 50)
    pScale = np.linspace(0.5, 1, 20)

    for P in pScale:
        print P
        infos.append(evaluateManyOccluders(min(n, m)*2, P, l, m, n, l, h, w, numSamples=200))

    p.plot(pScale, infos)
    ax = p.gca()

    ax.set_xlabel("Average permittivity of occluder")

#    for occluderN in occluderScale:

#        print occluderN
#        infos.append(evaluateManyOccluders(occluderN, 0.5, l, m, n, l, h, w, numSamples=200))

#    p.plot(occluderScale, infos)
#    p.axvline(x = 30)


#    ax.set_xlabel("Scale of occluder")
    ax.set_ylabel("Mutual Information (bits)")



    p.show()

if makeDarpaMetaPlots:

    wPlot = False
    mPlot = True

    if wPlot:

        pStars = []
        snrDBs = []
        pScale = np.linspace(0.5, 1, 21)
        logWScale = np.linspace(-8, 0, 17)

        for logW in logWScale:
            w = 10**logW
            bestP = None
            bestInfo = float("-inf")
            snr = cf*cf/w
            for P in pScale:

                info = evaluateManyOccluders(min(n, m)*2, P, l, m, n, l, h, w, numSamples=400)
                if info > bestInfo:
                    bestInfo = info
                    bestP = P
                print logW, P, info, bestP, snr, 10*log(snr, 10)

            pStars.append(bestP)
            snrDBs.append(10*log(snr, 10))

        p.plot(snrDBs, pStars)
        ax = p.gca()

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Optimal permittivity of occluder")

        p.show()


    if mPlot:

        kStars = []
        mns = []
        kScale = np.linspace(5, 100, 20)
        mnScale = np.linspace(12, 30, 15)

        for mn in mnScale:
#            w = 10**logW
            bestK = None
            bestInfo = float("-inf")
#            snr = cf*cf/w
            for k in kScale:

                info = evaluateManyOccludersWithPriors(k, P, l, int(mn), int(mn), l, h, w, c, numSamples=100)
                if info > bestInfo:
                    bestInfo = info
                    bestK = k
                print mn, k, info, bestK

            kStars.append(bestK)
#            snrDBs.append(10*log(snr, 10))

        p.plot(mnScale, kStars)
        ax = p.gca()

        ax.set_xlabel("Scene and Obs. Plane Resolution")
        ax.set_ylabel("Optimal Scale of Occluder Variation")

        p.show()
            #infos.append(evaluateManyOccluders(min(n, m)*2, P, l, m, n, l, h, w, numSamples=200))

if makeDarpaBasicPlots:
    occ = randomOccluderDiscrete(30, 0.75, l)
#    print occ

    displayOccluder(occ)
#    displayOccluder(PINSPECK_OCCLUDER)
#    displayOccluder(EDGE_OCCLUDER)

#    p.matshow(getTransferMatrix(occ, m, n, l, h))
#    p.colorbar()

#    p.show()

#    print turnTransitionsSequenceIntoArray(queryOccluderPoint, 10, 1)
#print getAverageOverRange(0.2, 0.5)
