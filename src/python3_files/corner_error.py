
from math import tan, atan2, sqrt, asin, pi, sin
import numpy as np
import matplotlib.pyplot as p
import scipy.io


dy = np.random.normal(0, 1e-3)
dx = np.random.normal(0, 1e-3)
theta = pi/3
r = 1

THETA_DISCRETE = 100
NUM_SAMPLES = 10000
FIND_STDEVS = False
PHI_DISCRETE = 999
GET_INCREASING_WHITENESS_PLOT = False
DEPTH_OFF_EXPERIMENT = True
DEPTH_OFF_EXPERIMENT_2 = False

def computePhi(dx, dy, theta, r):
    beta = atan2(dx, dy)
    maxLight = r - dy*tan(theta) + dx
    f = r - maxLight/2
    alpha = asin(sqrt(dx*dx + dy*dy)*sin(theta-beta)/f)
    gamma = pi - alpha - theta + beta
    phi = pi - gamma + beta

    return phi

def computeF(dx, dy, phi, r, theta):
    if phi <= theta:
        return r

    beta = atan2(dx, dy)
    alpha = pi - (theta-beta) - (pi/2 + beta + pi/2-phi)
#    print alpha

    f = sin(theta-beta)*sqrt(dx*dx+dy*dy)/sin(alpha)

    return f

def average(l):
    return sum(l)/len(l)

def variance(l):
    return average([i*i for i in l]) - average(l)**2

def stdev(l):
    #print l
    #print variance(l)
    return sqrt(variance(l))

if FIND_STDEVS:

    listOfMeans = []
    listOfStdevs = []
    listOfThetas = []


    for thetaLarge in range(2, THETA_DISCRETE-2):
        theta = thetaLarge*pi/2./THETA_DISCRETE

        print(thetaLarge, "/", THETA_DISCRETE)

        listOfPhiThetaDiffs = []

        for _ in range(NUM_SAMPLES):
            dy = np.random.normal(0, 1e-3)
            dx = np.random.normal(0, 1e-4)

            phi = computePhi(dx, dy, theta, r)
            phi = max(phi, 0)
            phi = min(phi, pi/2)
            #print phi
            #print theta

            listOfPhiThetaDiffs.append(phi-theta)

        listOfMeans.append(average(listOfPhiThetaDiffs))
        listOfStdevs.append(stdev(listOfPhiThetaDiffs))
        listOfThetas.append(theta)

    p.plot(listOfThetas, [i+j for i,j in zip(listOfMeans, listOfStdevs)], "r-")
    p.plot(listOfThetas, listOfMeans, "b-")
    p.plot(listOfThetas, [i-j for i,j in zip(listOfMeans, listOfStdevs)], "r-")
    p.xlabel("Theta")
    p.ylabel("Error")
    p.show()

#dx = 0.1
#dy = 0.2
#theta = pi/4
#r = 1

#print computePhi(dx, dy, theta, r)

if GET_INCREASING_WHITENESS_PLOT:
    dy = 0.2
    dx = 0.1
    listOfWhites = []
    listOfPhis = []

    for phiLarge in range(1, PHI_DISCRETE-1):
        phi = phiLarge*pi/2/PHI_DISCRETE


        f = computeF(dx, dy, phi, r, theta)
        listOfWhites.append(max(0, r-f))
        listOfPhis.append(phi)

    listOfControl = []

    dx = 0
    dy = 0

    for phiLarge in range(1, PHI_DISCRETE-1):
        phi = phiLarge*pi/2/PHI_DISCRETE

        f = computeF(dx, dy, phi, r, theta)

        listOfControl.append(max(0, r-f))

    p.plot(listOfPhis, listOfWhites, "r-")
    p.plot(listOfPhis, listOfControl, "b-")
    p.xlabel("Angle")
    p.ylabel("Intensity")
    p.show()

if DEPTH_OFF_EXPERIMENT:
    DISTANCE_BETWEEN_CORNERS = 1.

    yDistances = [1., 2., 3., 4.]
    X_DISTANCE_DISCRETE = 100
    NUM_SAMPLES = 1

    listOfMeans = []
    listOfStdevs = []
    listOfXDistances = []

    listOfListsOfMeans = [[] for _ in range(len(yDistances))]

#    dy1 = np.random.normal(0, 1e-3)
#    dx1 = np.random.normal(0, 1e-3)

    dx1 = 0.0
    dy1 = -0.02
    dx2 = 0.0
    dy2 = 0.02

#    dy2 = np.random.normal(0, 1e-3)
#    dx2 = np.random.normal(0, 1e-3)

    listOfListsOfTrueDepths = [[] for _ in range(len(yDistances))]

    listOfListsOfFalseDepths = [[] for _ in range(len(yDistances))]

    for xDistanceLarge in range(-3*X_DISTANCE_DISCRETE, int(3*X_DISTANCE_DISCRETE)):
        xDistance = xDistanceLarge/X_DISTANCE_DISCRETE

        print(xDistanceLarge, "/", X_DISTANCE_DISCRETE)

        listOfXDistances.append(xDistance)


        for i, yDistance in enumerate(yDistances):

            listOfDepthErrors = []

            for _ in range(NUM_SAMPLES):

                if xDistance < 0:
                    theta1 = atan2(-xDistance, yDistance)
                    theta2 = atan2(-xDistance + DISTANCE_BETWEEN_CORNERS, yDistance)

                    phi1 = computePhi(dx1, dy1, theta1, r)
                    phi2 = computePhi(dx2, dy2, theta2, r)

                    print(phi1, phi2)

                    trueDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 + theta1) + 1./tan(pi/2 - theta2))
                    fakeDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 + phi1) + 1./tan(pi/2 - phi2))

                if xDistance >= 0 and xDistance < 1:
                    theta1 = atan2(xDistance, yDistance)
                    theta2 = atan2(-xDistance + DISTANCE_BETWEEN_CORNERS, yDistance)

                    phi1 = computePhi(-dx1, dy1, theta1, r)
                    phi2 = computePhi(dx2, dy2, theta2, r)

                    print(phi1, phi2)

                    trueDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 - theta1) + 1./tan(pi/2 - theta2))
                    fakeDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 - phi1) + 1./tan(pi/2 - phi2))

                if xDistance >= 1:
                    theta1 = atan2(xDistance, yDistance)
                    theta2 = atan2(xDistance - DISTANCE_BETWEEN_CORNERS, yDistance)

                    phi1 = computePhi(-dx1, dy1, theta1, r)
                    phi2 = computePhi(-dx2, dy2, theta2, r)

                    print(phi1, phi2)

                    trueDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 - theta1) + 1./tan(pi/2 + theta2))
                    fakeDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 - phi1) + 1./tan(pi/2 + phi2))

                listOfDepthErrors.append(trueDepth - fakeDepth)

    #        listOfMeans.append(average(listOfDepthErrors))
            listOfListsOfMeans[i].append(average(listOfDepthErrors))
            listOfListsOfTrueDepths[i].append(trueDepth)
            listOfListsOfFalseDepths[i].append(fakeDepth)

        #    listOfStdevs.append(stdev(listOfDepthErrors))
    print([len(i) for i in listOfListsOfMeans])
    print(dx1, dy1, dx2, dy2)

#    p.plot(listOfXDistances, [i+j for i,j in zip(listOfMeans, listOfStdevs)], "r-")
#    p.plot(listOfXDistances, listOfMeans, "b-")
#    p.plot(listOfXDistances, [i-j for i,j in zip(listOfMeans, listOfStdevs)], "r-")
    for i, listOfMeans in enumerate(listOfListsOfMeans):
#        p.plot(listOfXDistances, listOfListsOfTrueDepths[i])
        p.plot(listOfXDistances, listOfListsOfFalseDepths[i])
        print(min(listOfMeans))
    p.xlabel("X Distance")
    p.ylabel("Error")
    p.show()

if DEPTH_OFF_EXPERIMENT_2:
    DISTANCE_BETWEEN_CORNERS = 1.

    yDistances = [1.,2.,3.,4.]
    X_DISTANCE_DISCRETE = 100
    NUM_SAMPLES = 10000

    listOfMeans = []
    listOfStdevs = []
    listOfXDistances = []

    listOfListsOfMeans = [[] for _ in range(len(yDistances))]

#    dx1 = 0.0
#    dy1 = -0.02
#    dx2 = 0.0
#    dy2 = 0.02


#    dy1 = np.random.normal(0, 1e-2)
#    dx1 = np.random.normal(0, 1e-2)

#    dy2 = np.random.normal(0, 1e-2)
#    dx2 = np.random.normal(0, 1e-2)

    listOfListsOfTrueDepths = [[] for _ in range(len(yDistances))]

    listOfListsOfFalseDepths = [[] for _ in range(len(yDistances))]

    listOfListsOfStdevs = [[] for _ in range(len(yDistances))]

    for xDistanceLarge in range(-1*X_DISTANCE_DISCRETE, int(2*X_DISTANCE_DISCRETE)):
        xDistance = xDistanceLarge/X_DISTANCE_DISCRETE

        print(xDistanceLarge, "/", X_DISTANCE_DISCRETE)

        listOfXDistances.append(xDistance)


        for i, yDistance in enumerate(yDistances):

            listOfDepthErrors = []
            listOfTrueDepths = []
            listOfFalseDepths = []

            for _ in range(NUM_SAMPLES):

                dy1 = np.random.normal(0, 1e-2)
                dx1 = np.random.normal(0, 1e-2)

                dy2 = np.random.normal(0, 1e-2)
                dx2 = np.random.normal(0, 1e-2)

                if xDistance < 0:
                    theta1 = atan2(-xDistance, yDistance)
                    theta2 = atan2(-xDistance + DISTANCE_BETWEEN_CORNERS, yDistance)

                    phi1 = computePhi(dx1, dy1, theta1, r)
                    phi2 = computePhi(dx2, dy2, theta2, r)

#                    print phi1, phi2

                    trueDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 + theta1) + 1./tan(pi/2 - theta2))
                    fakeDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 + phi1) + 1./tan(pi/2 - phi2))

                if xDistance >= 0 and xDistance < 1:
                    theta1 = atan2(xDistance, yDistance)
                    theta2 = atan2(-xDistance + DISTANCE_BETWEEN_CORNERS, yDistance)

                    phi1 = computePhi(-dx1, dy1, theta1, r)
                    phi2 = computePhi(dx2, dy2, theta2, r)

#                    print phi1, phi2

                    trueDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 - theta1) + 1./tan(pi/2 - theta2))
                    fakeDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 - phi1) + 1./tan(pi/2 - phi2))

                if xDistance >= 1:
                    theta1 = atan2(xDistance, yDistance)
                    theta2 = atan2(xDistance - DISTANCE_BETWEEN_CORNERS, yDistance)

                    phi1 = computePhi(-dx1, dy1, theta1, r)
                    phi2 = computePhi(-dx2, dy2, theta2, r)

#                    print phi1, phi2

                    trueDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 - theta1) + 1./tan(pi/2 + theta2))
                    fakeDepth = DISTANCE_BETWEEN_CORNERS/(1./tan(pi/2 - phi1) + 1./tan(pi/2 + phi2))

                listOfDepthErrors.append(trueDepth - fakeDepth)
                listOfTrueDepths.append(trueDepth)
                listOfFalseDepths.append(fakeDepth)

    #        listOfMeans.append(average(listOfDepthErrors))
            listOfListsOfMeans[i].append(average(listOfDepthErrors))
            listOfListsOfStdevs[i].append(stdev(listOfDepthErrors))
            listOfListsOfTrueDepths[i].append(average(listOfTrueDepths))
            listOfListsOfFalseDepths[i].append(average(listOfFalseDepths))

        #    listOfStdevs.append(stdev(listOfDepthErrors))
    print([len(i) for i in listOfListsOfMeans])
    print(dx1, dy1, dx2, dy2)

#    scipy.io.savemat("corner_error.mat", mdict={"means": \
#        np.array(listOfListsOfFalseDepths), "stdevs": \
#        np.array(listOfListsOfStdevs)})

    for i, listOfMeans in enumerate(listOfListsOfMeans):
        listOfStdevs = listOfListsOfStdevs[i]
        listOfFalseDepths = listOfListsOfFalseDepths[i]
        p.plot(listOfXDistances, [i+j for i,j in zip(listOfFalseDepths, listOfStdevs)], "r-")
        p.plot(listOfXDistances, listOfFalseDepths, "b-")
        p.plot(listOfXDistances, [i-j for i,j in zip(listOfFalseDepths, listOfStdevs)], "r-")
    p.xlabel("X Distance")
    p.ylabel("Error")
    p.show()
