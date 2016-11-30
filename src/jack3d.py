from __future__ import division
from math import sqrt, pi, sin, cos, cosh, sinh, acosh, asinh
import numpy as np
import matplotlib.pyplot as p
import random
from scipy.optimize import fmin_bfgs

TIME_STEP = 0.03
MIN_TIME = 1.
MAX_TIME = 20.

SPACE_EPS = 0.003
DIV0_EPS = 1e-7

TIME_LIGHT_ARRAY = [0.] * int(MAX_TIME/TIME_STEP - MIN_TIME/TIME_STEP)

def normalize(vec):
    return vec/np.linalg.norm(vec)

def stringIntersect(s1, s2):
    result = ""

    for char in s1:
        if char in s2:
            result += char

    return result

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        self.vec = np.array([x,y,z])

        self.personalTLA = [0.] * int(MAX_TIME/TIME_STEP - MIN_TIME/TIME_STEP)

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"

    @classmethod
    def vectorForm(cls, vec):
        return cls(vec[0], vec[1], vec[2])

    def distanceToPoint(self, otherPoint):
        return sqrt((self.x-otherPoint.x)*(self.x-otherPoint.x) \
                    +(self.y-otherPoint.y)*(self.y-otherPoint.y) \
                    +(self.z-otherPoint.z)*(self.z-otherPoint.z))

    def distanceToWall(self, wall):
        return abs(self.signedDistanceToWall(wall))

    def signedDistanceToWall(self, wall):
        return -(self.x*wall.a + self.y*wall.b + self.z*wall.c + wall.d) / \
                sqrt(wall.a*wall.a + wall.b*wall.b + wall.c*wall.c)

class LineSegment:
    def __init__(self, startLocation, stopLocation, normalVector):
        self.startLocation = startLocation
        self.stopLocation = stopLocation
        self.normalVector = normalVector
        self.startToStopVector = stopLocation - startLocation
        self.lineLength = np.linalg.norm(self.startToStopVector)

    def getListOfPoints(self):
        listOfPoints = []

        normalizedStartToStopVector = self.startToStopVector/self.lineLength
        numPoints = int(self.lineLength/SPACE_EPS)

        for pointNum in range(numPoints):
            pointVec = self.startLocation + pointNum * \
                SPACE_EPS * normalizedStartToStopVector

            listOfPoints.append(Point.vectorForm(pointVec))

        return listOfPoints

class Triangle:
    def __init__(self, v0, v1, v2):
        self.triangleDict = {}

        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

        self.triangleDict["0"] = self.v0
        self.triangleDict["1"] = self.v1
        self.triangleDict["2"] = self.v2

        x0 = v0[0]
        y0 = v0[1]
        z0 = v0[2]

        x1 = v1[0]
        y1 = v1[1]
        z1 = v1[2]

        x2 = v2[0]
        y2 = v2[1]
        z2 = v2[2]

        self.edge01 = self.v1 - self.v0
        self.edge02 = self.v2 - self.v0
        self.edge12 = self.v2 - self.v1

        self.triangleDict["01"] = self.edge01
        self.triangleDict["02"] = self.edge02
        self.triangleDict["12"] = self.edge12

        # edges of the triangle sorted by increasing length
        self.edgesByLength = sorted(["01", "02", "12"],
            key = lambda s: np.linalg.norm(self.triangleDict[s]))


        self.normalVector = normalize(np.cross(v1-v0, v2-v1))

        self.intercept = np.dot(self.v0, self.normalVector)

        assert abs(np.dot(self.v1, self.normalVector) - self.intercept) < DIV0_EPS
        assert abs(np.dot(self.v2, self.normalVector) - self.intercept) < DIV0_EPS

        self.planeConditions = [self.generatePlaneCondition(self.v0, self.v1, self.v2), \
            self.generatePlaneCondition(self.v2, self.v0, self.v1), \
            self.generatePlaneCondition(self.v1, self.v2, self.v0)]

    def __str__(self):
        return "Triangle with the following vertices: \n" + str(self.v0) + \
            + "\n" + str(self.v1) + "\n" + str(self.v2)

    def generatePlaneCondition(self, v0, v1, v2):

        planeNormal = np.cross(v1-v0, np.random.random(3))

        intercept = np.dot(planeNormal, v0)
        assert abs(np.dot(planeNormal, v1) - intercept) < DIV0_EPS

        resultOnV2 = np.dot(planeNormal, v2)

        if resultOnV2 >= intercept:
            print v0, v1, planeNormal, ">="
            return lambda v: np.dot(planeNormal, v) >= intercept

        elif resultOnV2 < intercept:
            print v0, v1, planeNormal, "<"
            return lambda v: np.dot(planeNormal, v) < intercept

        else:
            raise


    def testVectorForMembership(self, vec):

#        print np.dot(vec, self.normalVector), self.intercept

        if abs(np.dot(vec, self.normalVector) - self.intercept) < DIV0_EPS:
            for planeCondition in self.planeConditions:
                if not planeCondition(vec):
                    return False
            return True
        return False

    def getListOfPoints(self):
        listOfPoints = []

        e01 = self.triangleDict["01"]
        e02 = self.triangleDict["02"]

        v0 = self.triangleDict["0"]
        v2 = self.triangleDict["2"]

        e02ProjectionOntoE01 = e01 * np.dot(e01, e02) / \
            (np.linalg.norm(e01) * np.linalg.norm(e02))

        heightBase = v0 + e02ProjectionOntoE01
        heightVec = v2 - heightBase
        height = np.linalg.norm(heightVec)
        longLength = np.linalg.norm(e01)

        print self.normalVector

        print np.dot(heightVec, self.normalVector)
        print np.dot(e01, self.normalVector)

        adjustedHeight = int(height / SPACE_EPS)
        adjustedLongLength = int(longLength / SPACE_EPS)

        for longVectorLarge in range(adjustedLongLength):
#            print str(longVectorLarge) + " / " + str(adjustedLongLength)
            longVectorRegular = longVectorLarge*SPACE_EPS*normalize(e01)
            for heightVectorLarge in range(adjustedHeight):
                heightVectorRegular = heightVectorLarge*SPACE_EPS*normalize(heightVec)

                vec = v0 + longVectorRegular + heightVectorRegular
                if self.testVectorForMembership(vec):
                    listOfPoints.append(Point.vectorForm(vec))

        return listOfPoints

class Wall:
    def __init__(self, a, b, c, d, unitVector1, unitVector2,
        lambertianReflectance=1.0, specularReflectance=0.0, name="w"):
        # The wall is a plane characterized by the equation ax+by+cz=d
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        # The two unit vectors form a basis that spans the wall
        self.unitVector1 = np.array(unitVector1)
        self.unitVector2 = np.array(unitVector2)

        self.normalVectorMagnitude = sqrt(a*a + b*b + c*c)
        self.normalVector = np.array([a, b, c])/self.normalVectorMagnitude

        self.lambertianReflectance = lambertianReflectance
        self.specularReflectance = specularReflectance

        self.name = name

    def __str__(self):
        return "Wall " + self.name + ": " + str(self.a) + "x + " + \
            str(self.b) + "y + " + \
            str(self.c) + "z = " + \
            str(self.d)

    def getListOfPoints(self, squareRadius, examplePointOnWallInVectorForm):
        listOfPoints = []

        adjustedSquareRadius = int(squareRadius/SPACE_EPS)

        for uvFactor1Large in range(-adjustedSquareRadius, adjustedSquareRadius+1):
            uvFactor1 = uvFactor1Large*SPACE_EPS

            for uvFactor2Large in range(-adjustedSquareRadius, adjustedSquareRadius+1):
                uvFactor2 = uvFactor2Large*SPACE_EPS

                resultVector = uvFactor1*self.unitVector1 + \
                    uvFactor2*self.unitVector2 + \
                    examplePointOnWallInVectorForm

                listOfPoints.append(Point(resultVector[0], resultVector[1],
                    resultVector[2]))

        return listOfPoints

    def getRandomExampleVector(self):
#        randomVector = np.random.random(3)
        randomVector = np.array([ 0.11480026,  0.31741417,  0.27275809])

        return np.cross(randomVector, self.normalVector)

    def nearestPointOnWall(self, point):
        signedDistanceToWall = point.signedDistanceToWall(self)

        newPointVec = point.vec + self.normalVector * signedDistanceToWall

        return Point(newPointVec[0], newPointVec[1], newPointVec[2])

    def reflectThroughWall(self, point):
        signedDistanceToWall = point.signedDistanceToWall(self)

        newPointVec = point.vec + 2 * self.normalVector * signedDistanceToWall

        tla = point.personalTLA

        reflectedPoint = Point(newPointVec[0], newPointVec[1], newPointVec[2])
        reflectedPoint.personalTLA = tla

        return reflectedPoint

def convertTLAIndexToTime(tlaIndex):
    # Assumes MIN_TIME = 0

    return tlaIndex * TIME_STEP

def generateWall(normalVector, point):
    a = normalVector[0]
    b = normalVector[1]
    c = normalVector[2]
    d = a*point.x + b*point.y + c*point.z

#    randomVector = np.random.random(3)
    randomVector = np.array([ 0.61383678,  0.04041214,  0.34736099])
    spanVector1 = np.cross(randomVector, normalVector)
    unitVector1 = spanVector1 / np.linalg.norm(spanVector1)

    spanVector2 = np.cross(spanVector1, normalVector)
    unitVector2 = spanVector2 / np.linalg.norm(spanVector2)

    return Wall(a, b, c, d, unitVector1, unitVector2)

def addToTimeLightArray(time, lightAmount):
    timeIndex = int((time-MIN_TIME)/TIME_STEP)

    if timeIndex < len(TIME_LIGHT_ARRAY) and timeIndex > 0:
        TIME_LIGHT_ARRAY[timeIndex] += lightAmount

def initializeTimeLightArrayObject():
    return [0.] * int(MAX_TIME/TIME_STEP - MIN_TIME/TIME_STEP)

def addToTimeLightArrayObject(time, lightAmount, array):
    timeIndex = int((time-MIN_TIME)/TIME_STEP)

    if timeIndex < len(array) and timeIndex > 0:
        array[timeIndex] += lightAmount

def cumulativeAddToTimeLightArray(time, lightAmount):
    timeIndex = int((time-MIN_TIME)/TIME_STEP)

    for index in range(len(TIME_LIGHT_ARRAY)):
        if index >= timeIndex:
            TIME_LIGHT_ARRAY[index] += lightAmount

def plotTimeLightArray(filename):
    p.clf()
    p.plot([i*TIME_STEP for i in range(int(MIN_TIME/TIME_STEP), int(MAX_TIME/TIME_STEP))], TIME_LIGHT_ARRAY)
    p.savefig(filename)

def lightFactor(distance, comingFromWall=False, receivedByWall=False,
    distanceToWall=None):

    result = SPACE_EPS*SPACE_EPS/(4*pi*distance*distance)

    if receivedByWall:
        result *= distanceToWall/distance

    if comingFromWall:
        result *= 2

    return result

def timeOfLeg(sourcePoint, bouncePoint):
    return sourcePoint.distanceToPoint(bouncePoint)

# Light Factor functions for the primary experiment

def lightFactorWallToWall(sourcePoint, wallPoint, wall):
    return lightFactor(sourcePoint.distanceToPoint(wallPoint), True, True,
        sourcePoint.distanceToWall(wall))

def lightFactorPointToWall(sourcePoint, wallPoint, wall):
    return lightFactor(sourcePoint.distanceToPoint(wallPoint), False, True,
        sourcePoint.distanceToWall(wall))

def lightFactorWallToPoint(wallPoint, targetPoint):
    return lightFactor(wallPoint.distanceToPoint(targetPoint), comingFromWall=True,
        receivedByWall=False)

def lightFactorFirstLeg(sourcePoint, bouncePoint):
    return 1

def lightFactorSecondLegPointTarget(bouncePoint, targetPoint):
    return lightFactorWallToPoint(bouncePoint, targetPoint)

def lightFactorSecondLegWallTarget(bouncePoint, targetPoint, targetWall):
    return lightFactorWallToWall(bouncePoint, targetPoint, targetWall)

def lightFactorThirdLegPointTarget(targetPoint, wallPoint, wall):
    return lightFactorPointToWall(targetPoint, wallPoint, wall)

def lightFactorThirdLegWallTarget(targetPoint, wallPoint, wall):
    return lightFactorWallToWall(targetPoint, wallPoint, wall)

def lightFactorFourthLeg(wallPoint, detectorPoint):
    return lightFactorWallToPoint(wallPoint, detectorPoint)

def plotTimeLightArrayAndApproximation(approxFuncList, T, fileName):
    plotTimeLightArray(fileName)

    for approxTuple in approxFuncList:
        approxFunc = approxTuple[0]
        decoration = approxTuple[1]

        tlaApprox = []
        tApprox = []

        for rawTime in range(int(MIN_TIME/TIME_STEP),int(MAX_TIME/TIME_STEP)):
            t = rawTime*TIME_STEP-T

            tlaApprox.append(approxFunc(t))

            tApprox.append(T+t)

        p.plot(tApprox, tlaApprox, decoration)

    p.savefig(fileName)
    p.show()

def plotTimeLightArrayAndApproximationNoBigT(approxFuncList, fileName):
    plotTimeLightArray(fileName)

    for approxTuple in approxFuncList:
        approxFunc = approxTuple[0]
        decoration = approxTuple[1]

        tlaApprox = []
        tApprox = []

        for rawT in range(int(MIN_TIME/TIME_STEP),int(MAX_TIME/TIME_STEP)):
            t = rawT*TIME_STEP
            tlaApprox.append(approxFunc(t))
            tApprox.append(t)

        p.plot(tApprox, tlaApprox, decoration)

    p.savefig(fileName)
    p.show()

def plotArray(array):
    p.plot([i*TIME_STEP for i in range(int(MIN_TIME/TIME_STEP), int(MAX_TIME/TIME_STEP))], array, "b-")

def plotTimeLightArrayAndCandidate(candidateTLA):
    p.clf()
    plotArray(TIME_LIGHT_ARRAY)
    plotArray(candidateTLA)
    p.show()
    p.savefig("candidate.png")

def doSpherePointExperiment():
    T=1

    sourcePoint = Point(0,0,T)

    # Wall at z=0
    wall = Wall(0,0,1.,0, \
        np.array([1.,0,0]), \
        np.array([0,1.,0]))

    wallPoints = wall.getListOfPoints(5., np.array([0,0,0]))

    for wallPoint in wallPoints:
        time = timeOfLeg(sourcePoint, wallPoint)
        lightAmount = lightFactorPointToWall(sourcePoint, wallPoint, wall)

        addToTimeLightArray(time, lightAmount)
#        addToTimeLightArray(time, 1)

    def approxFunc(t):
        if t < 0.:
            return 0.

        return T*TIME_STEP/(2*(T+t)*(T+t))

    plotTimeLightArrayAndApproximation([(approxFunc, "r-")], T, "spherepoint.png")

def doSpherePointExperimentBothWays():
    T=1

    sourcePoint = Point(0,0,T)

    # Wall at z=0
    wall = Wall(0,0,1.,0, \
        np.array([1.,0,0]), \
        np.array([0,1.,0]))

    wallPoints = wall.getListOfPoints(5., np.array([0,0,0]))

    for wallPoint in wallPoints:
        time = timeOfLeg(sourcePoint, wallPoint)
        lightAmount = lightFactorWallToWall(sourcePoint, wallPoint, wall)

        addToTimeLightArray(time, lightAmount**2)
        # squaring it because of the both legs

#        addToTimeLightArray(time, 1)

    def approxFunc(t):
        if t < 0.:
            return 0.

        return T*TIME_STEP/((T+t)*(T+t)) * (T*SPACE_EPS*SPACE_EPS)/ \
            (2*pi*(T+t)*(T+t)*(T+t))

    plotTimeLightArrayAndApproximation([(approxFunc, "r-")], T, "spherepoint.png")

def doDoubleWallExperiment():
    T=2.

    sourcePoint = Point(0,0,0)

    # Wall at z=0
    detectorWall = Wall(0,0,1.,0, \
        np.array([1.,0,0]), \
        np.array([0.,1.,0]))

    # Wall at z=T/2
    reflectorWall = Wall(0,0,1.,T/2., \
        np.array([1.,0,0]), \
        np.array([0.,1.,0]))

    reflectorWallPoints = reflectorWall.getListOfPoints(2., np.array([0,0,T/2.]))
    detectorWallPoints = detectorWall.getListOfPoints(2., np.array([0,0,0]))

    for i, reflectorWallPoint in enumerate(reflectorWallPoints):
        for detectorWallPoint in detectorWallPoints:

            time = timeOfLeg(sourcePoint, reflectorWallPoint) + \
                    timeOfLeg(reflectorWallPoint, detectorWallPoint)

            lightAmount = lightFactorWallToWall(sourcePoint, reflectorWallPoint, reflectorWall) * \
                lightFactorWallToWall(reflectorWallPoint, detectorWallPoint, detectorWall)

            addToTimeLightArray(time, lightAmount)

    plotTimeLightArray()

    tlaApprox = []
    tApprox = []

    for rawTime in range(int(MIN_TIME/TIME_STEP),int(MAX_TIME/TIME_STEP)):
        t = rawTime*TIME_STEP-T
        if t<0:
            tlaApprox.append(0.)
        else:
            tlaApprox.append(17./6.*((t+TIME_STEP)/T)**(3./2.)*(TIME_STEP))

        tApprox.append(T+t)

    p.plot(tApprox, tlaApprox)
    p.savefig("tla_approx.png")


# points are at (-d, 0, 0) (source) and (d, 0, 0) (detector)
def doTwoPointPlaneExperimentWallDetect(d, T, nu, phi):

    observationArray = [0.] * int(MAX_TIME/TIME_STEP - MIN_TIME/TIME_STEP)

    reflectorWall = Wall(0.,0.,1.,0., \
        np.array([1.,0.,0.]),
        np.array([0.,1.,0.]))


    sourcePoint = Point(-d, 0, 0)
    detectorPoint = Point(d, 0, 0)

    targetWall = getPlaneTangentTo(d, T, nu, phi)

    targetNormalVector = targetWall.normalVector


    targetWallPoints = targetWall.getListOfPoints(10., findSpotOnEggshell(d, T, nu, phi).vec)

    totalGlanceFactorsArray = initializeTimeLightArrayObject()
    numGlanceFactorsArray = initializeTimeLightArrayObject()



    for i, wallPoint in enumerate(targetWallPoints):
        time = timeOfLeg(sourcePoint, wallPoint) + \
            timeOfLeg(wallPoint, detectorPoint)

        lightAmount = lightFactorPointToWall(sourcePoint, wallPoint, targetWall) * \
            lightFactorWallToWall(wallPoint, detectorPoint, reflectorWall)



        glanceFactor = lightFactorWallToWall(wallPoint, detectorPoint, reflectorWall) / \
                lightFactorWallToPoint(wallPoint, detectorPoint)

        # DEBUG
        addToTimeLightArrayObject(time, glanceFactor, totalGlanceFactorsArray)
        addToTimeLightArrayObject(time, 1., numGlanceFactorsArray)
        # END DEBUG

        addToTimeLightArrayObject(time, lightAmount, observationArray)

    averageGlanceFactorsArray = [i/(j+DIV0_EPS) for i, j in zip(totalGlanceFactorsArray, numGlanceFactorsArray)]

#    p.clf()
#    plotArray(averageGlanceFactorsArray)
#    p.show()
#    p.clf()

    model, modelGlanceFactors = candidateTLAMaker(d, T)(np.array([nu, phi]))

    plotArray(observationArray)

    plotArray(model)

    p.show()
#    p.clf()
#    observationsWithoutGlanceFactor = [i/(j+DIV0_EPS)*(j!=0) for i, j in zip(observationArray, averageGlanceFactorsArray)]
#    modelWithoutGlanceFactor = [i/(j+DIV0_EPS)*(j!=0) for i, j in zip(model, modelGlanceFactors)]

#    plotArray(observationsWithoutGlanceFactor)
#    plotArray(modelWithoutGlanceFactor)
#    p.show()

    # Now I just need to run the experiment with my experiment-running functions


def doTwoPointPlaneExperiment():
    # xs = 0. (ASSUMED WLOG)
    # ys = 0. (ASSUMED WLOG)
    zs = 1. #+0.5*SPACE_EPS

    xd = 1. #+0.5*SPACE_EPS
    # yd = 0. (ASSUMED WLOG)
    zd = 1. #+0.5*SPACE_EPS

    x = xd*(zs/(zd+zs))

    sourcePoint = Point(0.,0.,zs)
    detectorPoint = Point(xd,0.,zd)

    # We know the location of the target because we did a knowledge inversion
    # z = 0
    targetWall = Wall(0.,0.,1.,0., \
        np.array([1.,0.,0.]),
        np.array([0.,1.,0.]))

    T = timeOfLeg(sourcePoint, targetWall.reflectThroughWall(detectorPoint))

    targetWallPoints = targetWall.getListOfPoints(10., np.array([x,0.,0.]))

    for i, wallPoint in enumerate(targetWallPoints):
        time = timeOfLeg(sourcePoint, wallPoint) + \
            timeOfLeg(wallPoint, detectorPoint)

        lightAmount = lightFactorPointToWall(sourcePoint, wallPoint, targetWall) * \
            lightFactorWallToPoint(wallPoint, detectorPoint)

#        cumulativeAddToTimeLightArray(time, 1)
#        cumulativeAddToTimeLightArray(time, lightAmount)
#        addToTimeLightArray(time, 1)
        addToTimeLightArray(time, lightAmount)

        if i % 10000 == 0:
            print i

    def aMinor(t):
        xdm2x = xd - 2*x

        num1 = (t+T)*(t+T)*xdm2x
        num2 = -xd*(xd*xdm2x + zd*zd - zs*zs)

        denom = 2*(t+T-xd)*(t+T+xd)

        return (num1+num2)/denom

    def a(t):
        # Assumes t >= 0
        # Born of a Mathematica result

        tt2 = (t+T)*(t+T)

        radical1 = (-tt2 + xd*xd + zd*zd)**2
        radical2 = -2*(tt2 - xd*xd + zd*zd)*zs*zs
        radical3 = zs**4

        num = sqrt(tt2*(radical1 + radical2 + radical3))

        denom = 2*(t+T-xd)*(t+T+xd)

        return num/denom

    def bouncePoint(t):
        return x + aMinor(t)

    def b(t, xb):
        # Assumes t >= 0
        # Born of a Mathematica result

        rad1 = t**4 + 4*t**3*T + T**4
        rad2 = 4*t*T*(T*T - 2*xb*xb + 2*xb*xd - xd*xd - zd*zd - zs*zs)
        rad3 = (-2*xb*xd + xd*xd + zd*zd - zs*zs)**2
        rad4 = -2*T*T*(2*xb*xb - 2*xb*xd + xd*xd + zd*zd + zs*zs)
        rad5 = t*t*(6*T*T - 2*(2*xb*xb - 2*xb*xd + xd*xd + zd*zd + zs*zs))

        return sqrt(rad1+rad2+rad3+rad4+rad5)/(2*(t+T))

    # TODO: Remember nexts if you decide you want to speed this up
    # by a factor of 2
    def numPaths(aOfT, bOfT, dadt, dbdt):
        return pi/(SPACE_EPS*SPACE_EPS)*(aOfT*dbdt \
            + dadt*bOfT)

    def pathStrengthBSourceToWall(xb, bOfT):
        return zs*SPACE_EPS*SPACE_EPS/ \
            (4*pi*(xb*xb + bOfT*bOfT + zs*zs)**1.5)

    def pathStrengthBWallToDetector(xb, bOfT):
        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-xd)**2 + \
            bOfT*bOfT + zd*zd))

    def pathStrengthCenterSourceToWall(xb, bOfT):
        return zs*SPACE_EPS*SPACE_EPS/ \
            (4*pi*(xb*xb + zs*zs)**1.5)

    def pathStrengthCenterWallToDetector(xb, bOfT):
        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-xd)**2 + \
            zd*zd))

    def pathStrengthAMinusSourceToWall(xb, aOfT):
        return zs*SPACE_EPS*SPACE_EPS / \
            (4*pi*((xb-aOfT)**2 + zs*zs)**1.5)

    def pathStrengthAMinusWallToDetector(xb, aOfT):
        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-aOfT-xd)**2 + \
            zd*zd))

    def pathStrengthAPlusSourceToWall(xb, aOfT):
        return zs*SPACE_EPS*SPACE_EPS / \
            (4*pi*((xb+aOfT)**2 + zs*zs)**1.5)

    def pathStrengthAPlusWallToDetector(xb, aOfT):
        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-aOfT-xd)**2 + \
            zd*zd))

    def pathStrengthCenter(xb, bOfT):
        return pathStrengthCenterSourceToWall(xb, bOfT) * \
                pathStrengthCenterWallToDetector(xb, bOfT)

    def pathStrengthB(xb, bOfT):
        return pathStrengthBSourceToWall(xb, bOfT) * \
                pathStrengthBWallToDetector(xb, bOfT)

    def pathStrengthAMinus(xb, aOfT):
        return pathStrengthAMinusSourceToWall(xb, aOfT) * \
                pathStrengthAMinusWallToDetector(xb, aOfT)

    def pathStrengthAPlus(xb, aOfT):
        return pathStrengthAPlusSourceToWall(xb, aOfT) * \
                pathStrengthAPlusWallToDetector(xb, aOfT)

    def pathStrengthWeightAverage(xb, aOfT, bOfT):
        return 0.5*pathStrengthB(xb, bOfT) + \
                0.25*(pathStrengthAMinus(xb, aOfT) + \
                    pathStrengthAPlus(xb, aOfT))

    def approxFuncB(prevT):
        t = prevT + TIME_STEP

        if prevT < 0.:
            return 0.

        xb = bouncePoint(t)
        nextXb = bouncePoint(t+TIME_STEP)

        aOfT = a(t)
        bOfT = b(t, xb)

        aOfNextT = a(t+TIME_STEP)
        bOfNextT = b(t+TIME_STEP, nextXb)

        dadt = aOfNextT-aOfT
        dbdt = bOfNextT-bOfT

        return numPaths(aOfT, bOfT, dadt, dbdt)*pathStrengthB(xb, bOfT)

    def approxFuncAMinus(prevT):
        t = prevT + TIME_STEP

        if prevT < 0.:
            return 0.

        xb = bouncePoint(t)
        nextXb = bouncePoint(t+TIME_STEP)

        aOfT = a(t)
        bOfT = b(t, xb)

        aOfNextT = a(t+TIME_STEP)
        bOfNextT = b(t+TIME_STEP, nextXb)

        dadt = aOfNextT-aOfT
        dbdt = bOfNextT-bOfT

        return numPaths(aOfT, bOfT, dadt, dbdt)*pathStrengthAMinus(xb, aOfT)

    def approxFuncAPlus(prevT):
        t = prevT + TIME_STEP

        if prevT < 0.:
            return 0.

        xb = bouncePoint(t)
        nextXb = bouncePoint(t+TIME_STEP)

        aOfT = a(t)
        bOfT = b(t, xb)

        aOfNextT = a(t+TIME_STEP)
        bOfNextT = b(t+TIME_STEP, nextXb)

        dadt = aOfNextT-aOfT
        dbdt = bOfNextT-bOfT

        return numPaths(aOfT, bOfT, dadt, dbdt)*pathStrengthAPlus(xb, aOfT)

    def approxFuncWeightAverage(prevT):
        return 0.5*approxFuncB(prevT) + \
            0.25*(approxFuncAMinus(prevT) + approxFuncAPlus(prevT))

    def approxFuncAverageA(prevT):
        return 0.5*(approxFuncAMinus(prevT) + approxFuncAPlus(prevT))

    def approxFunc(prevT):
        t = prevT + TIME_STEP

        if prevT < 0.:
            return 0.

        xb = bouncePoint(t)
        nextXb = bouncePoint(t+TIME_STEP)

        aOfT = a(t)
        bOfT = b(t, xb)

        aOfNextT = a(t+TIME_STEP)
        bOfNextT = b(t+TIME_STEP, nextXb)

        dadt = aOfNextT-aOfT
        dbdt = bOfNextT-bOfT

        return numPaths(aOfT, bOfT, dadt, dbdt)*pathStrengthWeightAverage(xb, aOfT, bOfT)




#    plotTimeLightArray("two_point_plane_cumulative.png")

#    plotTimeLightArrayAndApproximation(
#        [(approxFuncB, "r-"),
#        (approxFuncAMinus, "g-"),
#        (approxFuncAPlus, "c-")], T,
#        "two_point_plane.png")

    plotTimeLightArrayAndApproximation([(approxFunc, "r-")],
        T, "two_point_plane.png")

def approxFuncMaker():
    def aMinor(params, x, T, t):
        zs, xd, zd = params

        xdm2x = xd - 2*x

        num1 = (t+T)*(t+T)*xdm2x
        num2 = -xd*(xd*xdm2x + zd*zd - zs*zs)

        denom = 2*(t+T-xd)*(t+T+xd)

        return (num1+num2)/denom

    def a(params, T, t):
        # Assumes t >= 0
        # Born of a Mathematica result

        zs, xd, zd = params

        tt2 = (t+T)*(t+T)

        radical1 = (-tt2 + xd*xd + zd*zd)**2
        radical2 = -2*(tt2 - xd*xd + zd*zd)*zs*zs
        radical3 = zs**4

        num = sqrt(tt2*(radical1 + radical2 + radical3))

        denom = 2*(t+T-xd)*(t+T+xd)

        return num/denom

    def bouncePoint(params, T, t):
        zs, xd, zd = params

        x = xd*(zs/(zd+zs))

        return x + aMinor(params, x, T, t)

    def b(params, T, t, xb):
        # Assumes t >= 0
        # Born of a Mathematica result

        zs, xd, zd = params

        rad1 = t**4 + 4*t**3*T + T**4
        rad2 = 4*t*T*(T*T - 2*xb*xb + 2*xb*xd - xd*xd - zd*zd - zs*zs)
        rad3 = (-2*xb*xd + xd*xd + zd*zd - zs*zs)**2
        rad4 = -2*T*T*(2*xb*xb - 2*xb*xd + xd*xd + zd*zd + zs*zs)
        rad5 = t*t*(6*T*T - 2*(2*xb*xb - 2*xb*xd + xd*xd + zd*zd + zs*zs))

        return sqrt(rad1+rad2+rad3+rad4+rad5)/(2*(t+T))

    # TODO: Remember nexts if you decide you want to speed this up
    # by a factor of 2
    def numPaths(aOfT, bOfT, dadt, dbdt):
        return pi/(SPACE_EPS*SPACE_EPS)*(aOfT*dbdt \
            + dadt*bOfT)

    def pathStrengthBSourceToWall(params, xb, bOfT):
        zs, xd, zd = params

        return zs*SPACE_EPS*SPACE_EPS/ \
            (4*pi*(xb*xb + bOfT*bOfT + zs*zs)**1.5)

    def pathStrengthBWallToDetector(params, xb, bOfT):
        zs, xd, zd = params

        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-xd)**2 + \
            bOfT*bOfT + zd*zd))

    # Need to dot with the z = 0 plane normal vector
    def pathStrengthBMinusWallToDetectorWall(params, xb, bOfT, normalVec):
        zs, xd, zd = params

        wallPointToDetectorVec = np.array([xd-xb, -bOfT, zd])

        glanceFactor = abs(np.dot(wallPointToDetectorVec, normalVec) / \
            np.linalg.norm(wallPointToDetectorVec))

        return SPACE_EPS*SPACE_EPS*glanceFactor / (2*pi*((xb-xd)**2 + \
            bOfT*bOfT + zd*zd)), glanceFactor

    def pathStrengthBPlusWallToDetectorWall(params, xb, bOfT, normalVec):
        zs, xd, zd = params

        wallPointToDetectorVec = np.array([xd-xb, bOfT, zd])

        glanceFactor = abs(np.dot(wallPointToDetectorVec, normalVec) / \
            np.linalg.norm(wallPointToDetectorVec))

        return SPACE_EPS*SPACE_EPS*glanceFactor / (2*pi*((xb-xd)**2 + \
            bOfT*bOfT + zd*zd)), glanceFactor

    def pathStrengthCenterSourceToWall(params, xb, bOfT):
        zs, xd, zd = params

        return zs*SPACE_EPS*SPACE_EPS/ \
            (4*pi*(xb*xb + zs*zs)**1.5)

    def pathStrengthCenterWallToDetector(params, xb, bOfT):
        zs, xd, zd = params

        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-xd)**2 + \
            zd*zd))

    def pathStrengthCenterWallToDetectorWall(params, xb, bOfT, normalVec):
        zs, xd, zd = params

        wallPointToDetectorVec = np.array([xd-xb, 0, zd])

        glanceFactor = abs(np.dot(wallPointToDetectorVec, normalVec) / \
            np.linalg.norm(wallPointToDetectorVec))

        return SPACE_EPS*SPACE_EPS*glanceFactor / (2*pi*((xb-xd)**2 + \
            zd*zd))


    def pathStrengthAMinusSourceToWall(params, xb, aOfT):
        zs, xd, zd = params

        return zs*SPACE_EPS*SPACE_EPS / \
            (4*pi*((xb-aOfT)**2 + zs*zs)**1.5)

    def pathStrengthAMinusWallToDetector(params, xb, aOfT):
        zs, xd, zd = params

        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-aOfT-xd)**2 + \
            zd*zd))

    def pathStrengthAMinusWallToDetectorWall(params, xb, aOfT, normalVec):
        zs, xd, zd = params

        wallPointToDetectorVec = np.array([xd+aOfT-xb, 0, zd])

        glanceFactor = abs(np.dot(wallPointToDetectorVec, normalVec) / \
            np.linalg.norm(wallPointToDetectorVec))

        return SPACE_EPS*SPACE_EPS*glanceFactor / (2*pi*((xb-aOfT-xd)**2 + \
            zd*zd)), glanceFactor

    def pathStrengthAPlusSourceToWall(params, xb, aOfT):
        zs, xd, zd = params

        return zs*SPACE_EPS*SPACE_EPS / \
            (4*pi*((xb+aOfT)**2 + zs*zs)**1.5)

    def pathStrengthAPlusWallToDetector(params, xb, aOfT, normalVec):
        zs, xd, zd = params

        return SPACE_EPS*SPACE_EPS / (2*pi*((xb+aOfT-xd)**2 + \
            zd*zd))

    def pathStrengthAPlusWallToDetectorWall(params, xb, aOfT, normalVec):
        zs, xd, zd = params

        wallPointToDetectorVec = np.array([xd-aOfT-xb, 0, zd])

        glanceFactor = abs(np.dot(wallPointToDetectorVec, normalVec) / \
            np.linalg.norm(wallPointToDetectorVec))

#        print glanceFactor,  Point(xb+aOfT, 0, 0), Point(xd, 0, zd)

        return SPACE_EPS*SPACE_EPS*glanceFactor / (2*pi*((xb+aOfT-xd)**2 + \
            zd*zd)), glanceFactor


    def pathStrengthCenter(params, xb, bOfT, normalVec):
        return pathStrengthCenterSourceToWall(params, xb, bOfT) * \
                pathStrengthCenterWallToDetectorWall(params, xb, bOfT, normalVec)

    def pathStrengthBMinus(params, xb, bOfT, normalVec):
        psb, gf = pathStrengthBMinusWallToDetectorWall(params, xb, bOfT, normalVec)

        return pathStrengthBSourceToWall(params, xb, bOfT) * psb, gf

    def pathStrengthBPlus(params, xb, bOfT, normalVec):
        psb, gf = pathStrengthBPlusWallToDetectorWall(params, xb, bOfT, normalVec)

        return pathStrengthBSourceToWall(params, xb, bOfT) * psb, gf

    def pathStrengthAMinus(params, xb, aOfT, normalVec):
        psa, gf = pathStrengthAMinusWallToDetectorWall(params, xb, aOfT, normalVec)

        return pathStrengthAMinusSourceToWall(params, xb, aOfT) * psa, gf

    def pathStrengthAPlus(params, xb, aOfT, normalVec):
        psa, gf = pathStrengthAPlusWallToDetectorWall(params, xb, aOfT, normalVec)

        return pathStrengthAPlusSourceToWall(params, xb, aOfT) * psa, gf

    def pathStrengthWeightAverage(params, xb, aOfT, bOfT, normalVec):
        psBm, gfBm = pathStrengthBMinus(params, xb, bOfT, normalVec)
        psBp, gfBp = pathStrengthBPlus(params, xb, bOfT, normalVec)
        psAm, gfAm = pathStrengthAMinus(params, xb, aOfT, normalVec)
        psAp, gfAp = pathStrengthAPlus(params, xb, aOfT, normalVec)

        gfOverall = 0.25*(gfBm + gfBp) + 0.25*(gfAm + gfAp)

        return 0.25*(psBm + psBp + psAm + psAp), gfOverall, gfBm, gfBp, gfAm, gfAp

    def approxFunc(params, T, prevT, normalVec, modelGlanceFactorTLA, gfBmArray, gfBpArray, gfAmArray, gfApArray):
        t = prevT + TIME_STEP

        if prevT < 0.:
            return 0.

        xb = bouncePoint(params, T, t)
        nextXb = bouncePoint(params, T, t+TIME_STEP)

        aOfT = a(params, T, t)
        bOfT = b(params, T, t, xb)

        aOfNextT = a(params, T, t+TIME_STEP)
        bOfNextT = b(params, T, t+TIME_STEP, nextXb)

        dadt = aOfNextT-aOfT
        dbdt = bOfNextT-bOfT

        pswa, gfo, gfBm, gfBp, gfAm, gfAp = pathStrengthWeightAverage(params, xb, aOfT, bOfT, normalVec)

        addToTimeLightArrayObject(t+T, gfo, modelGlanceFactorTLA)
        addToTimeLightArrayObject(t+T, gfBm, gfBmArray)
        addToTimeLightArrayObject(t+T, gfBp, gfBpArray)
        addToTimeLightArrayObject(t+T, gfAm, gfAmArray)
        addToTimeLightArrayObject(t+T, gfAp, gfApArray)

        return numPaths(aOfT, bOfT, dadt, dbdt)*pswa

    def firstLightFunc(params, targetWall):
        zs, xd, zd = params

        sourcePoint = Point(0.,0.,zs)
        detectorPoint = Point(xd,0.,zd)

        return timeOfLeg(sourcePoint, targetWall.reflectThroughWall(detectorPoint))

    return firstLightFunc, approxFunc

#def getValsFromApproxFunc(params, approxFunc):
#    T = firstLightFunc(params)
#
#    funcOfTPlus12 = approxFunc(params, T, 11.5*TIME_STEP)
#    funcOfTPlus15 = approxFunc(params, T, 14.5*TIME_STEP)

#    return (T, funcOfTPlus12, funcOfTPlus15-funcOfTPlus12)

def getFirstNonzeroIndexFromTLA():
    for i, val in enumerate(TIME_LIGHT_ARRAY):
        if val != 0.:
            return i

def extractTFromTLA():
    nonzeroIndex = getFirstNonzeroIndexFromTLA()

    T = MIN_TIME + (nonzeroIndex+0.5)*TIME_STEP

    return T


# Assume MIN_TIME = 0
def getCandidateTLA(params, targetWall, T, normalVec, approxFunc):
    lenTLA = len(TIME_LIGHT_ARRAY)

    numZeroes = int(T/TIME_STEP)

    tla = [0] * numZeroes

    modelGlanceFactorTLA = initializeTimeLightArrayObject()
    gfBmArray = initializeTimeLightArrayObject()
    gfBpArray = initializeTimeLightArrayObject()
    gfAmArray = initializeTimeLightArrayObject()
    gfApArray = initializeTimeLightArrayObject()

    tla += [approxFunc(params, T, (i+0.5)*TIME_STEP, normalVec, modelGlanceFactorTLA, \
        gfBmArray, gfBpArray, gfAmArray, gfApArray) for i in range(int(MAX_TIME/TIME_STEP) - numZeroes)]

#    plotArray(modelGlanceFactorTLA)
#    plotArray(gfBmArray)
#    plotArray(gfBpArray)
#    plotArray(gfAmArray)
#    plotArray(gfApArray)
#    p.show()
#    p.clf()

    return tla, modelGlanceFactorTLA

# normal vector is oriented
def bounceFactorOriented(sourceVec, targetVec, normalVec):
    incomingVector = targetVec - sourceVec
    incVecLength = np.linalg.norm(incomingVector)
    areaFraction = SPACE_EPS*SPACE_EPS/(2*pi*incVecLength*incVecLength)
    glanceFactor = -np.dot(incomingVector / incVecLength, normalVec)
    return max(glanceFactor*areaFraction, 0)

def bounceFactor(sourceVec, targetVec, normalVec):
    incomingVector = targetVec - sourceVec
    incVecLength = np.linalg.norm(incomingVector)
    areaFraction = SPACE_EPS*SPACE_EPS/(2*pi*incVecLength*incVecLength)
    glanceFactor = -np.dot(incomingVector / incVecLength, normalVec)
    return abs(glanceFactor*areaFraction)

def computeK(xd, v0, v1, t):
    x0 = v0[0]
    x1 = v1[0]
    y0 = v0[1]
    y1 = v1[1]
    z0 = v0[2]
    z1 = v1[2]

    xDiff = x1-x0
    yDiff = y1-y0
    zDiff = z1-z0

    xDestDiff = 2*x0-xd
    xd2x0xd = xd*xDestDiff

    xDiffProd = xDiff*xDestDiff
    yDiffProd = yDiff*2*y0
    zDiffProd = zDiff*2*z0

    normSquared = xDiff*xDiff + yDiff*yDiff + zDiff*zDiff
    tSquaredNormSquared = t*t*normSquared

    # (x0 - x1) (2 x0 - xd) xd^2
    num1 = -xDiffProd*xd*xd
    # t^2 ((-x0 + x1) (2 x0 - xd) + 2 y0 (-y0 + y1) + 2 z0 (-z0 + z1))
    num2 = t*t*(xDiffProd+yDiffProd+zDiffProd)
    numSum = num1+num2

    # t^4 + (-2 x0 xd + xd^2)^2
    rad1Prod1 = t**4 + xd2x0xd*xd2x0xd
    # 2 t^2 (2 x0^2 - 2 x0 xd + xd^2 + 2 (y0^2 + z0^2))
    rad2Prod1 = -2*t*t*(2*x0*x0 - xd2x0xd + 2*(y0*y0 + z0*z0))
    # -(x0 - x1)^2 xd^2
    rad1Prod2 = -xDiff*xDiff*xd*xd
    radProd2 = rad1Prod2+tSquaredNormSquared

    radProd = (rad1Prod1+rad2Prod1) * radProd2

    # (x0 - x1) (2 x0 - xd) xd^2
#            rad1Square = num1
    radSquare = numSum*numSum

    num = numSum + sqrt(radProd + radSquare)

    return num / (2*radProd2)

def getPointFromK(k, v0, v1):
    return v0 + k*(v0-v1)

def getDistanceBetweenPoints(v0, v1):
    return np.linalg.norm(v0-v1)

def lineCumNumPaths(xd, v0, v1):
    def approxFunc(t):
        source = np.array([0,0,0])
        detector = np.array([xd,0,0])

        startTime = getDistanceBetweenPoints(v0, source) + \
            getDistanceBetweenPoints(v0, detector)

        if t < startTime:
            return 0.
        else:
            k = computeK(xd, v0, v1, t)
            dist = getDistanceBetweenPoints(v0, getPointFromK(k, v0, v1))
            return dist/SPACE_EPS
    return approxFunc

def computeCumNumPathsFromK(k, xd, v0, v1):
    dist = getDistanceBetweenPoints(v0, getPointFromK(k, v0, v1))
    return dist/SPACE_EPS

def lineCumNumPathsOld(xd, v0, v1):
    def approxFunc(t):
        x0 = v0[0]
        x1 = v1[0]

        norm = np.linalg.norm(v0-v1)

        tMinusTPlus = (t-xd)*(t+xd)
        x0x1xd = (x0-x1)*xd
        nonRadNum = x0x1xd*tMinusTPlus
        tSquaredNormSquared = t*t*norm*norm
        radNum = tMinusTPlus*tMinusTPlus*tSquaredNormSquared
        denom = 2*(tSquaredNormSquared - x0x1xd*x0x1xd)

        return (nonRadNum + sqrt(radNum)) * norm / (denom*SPACE_EPS)
    return approxFunc

def lineNumPathsOld(xd, v0, v1):
    def approxFunc(t):
        x0 = v0[0]
        y0 = v0[1]
        z0 = v0[2]

        x1 = v1[0]
        y1 = v1[1]
        z1 = v1[2]

        norm = np.linalg.norm(v0-v1)
        x0x1xd = (x0-x1)*xd
        tSquaredNormSquared = t*t*norm*norm
        yzNormSquared = (y0-y1)**2+(z0-z1)**2

        num1 = -2 * t * x0x1xd * xd*xd * yzNormSquared
        num2 = -x0x1xd*x0x1xd * xd*xd * norm
        num3 = t*t * xd*xd * (2*(x0-x1)**2 - yzNormSquared) * norm
        num4 = -tSquaredNormSquared * t*t * norm

        denom = 2 * (x0x1xd*x0x1xd - tSquaredNormSquared)**2

        return -(num1 + num2 + num3 + num4) * norm / (denom * SPACE_EPS)
    return approxFunc


def lineNumPathsNumericDeriv(xd, v0, v1):
    def approxFunc(t):
        cumFunc = lineCumNumPaths(xd, v0, v1)
        return (cumFunc(t+TIME_STEP) - cumFunc(t))/TIME_STEP
    return approxFunc

def lineResponse(xd, v0, v1, lineNormal, detectorNormal):
    def approxFunc(t):
        source = np.array([0,0,0])
        detector = np.array([xd,0,0])

        startTime = getDistanceBetweenPoints(v0, source) + \
            getDistanceBetweenPoints(v0, detector)

        if t-TIME_STEP < startTime:
            return 0.

        kPrevious = computeK(xd, v0, v1, t-TIME_STEP)
        k = computeK(xd, v0, v1, t)

        if k > 1.:
            return 0.

        previousPointFromK = getPointFromK(kPrevious, v0, v1)
        pointFromK = getPointFromK(k, v0, v1)

        numPaths = (getDistanceBetweenPoints(v0, pointFromK) - \
            getDistanceBetweenPoints(v0, previousPointFromK)) / SPACE_EPS

        leg1ipp = bounceFactor(source, pointFromK, lineNormal)
        leg2ipp = bounceFactor(pointFromK, detector, detectorNormal)

        return numPaths * leg1ipp * leg2ipp

    return approxFunc

def triangleResponse(xd, v0, v1, v2, detectorNormal):
    def approxFunc(t):
        source = np.array([0,0,0])
        detector = np.array([xd,0,0])

def doLineExperiment():
    # xs = 0. (ASSUMED WLOG)
    # ys = 0. (ASSUMED WLOG)
    # zs = 0. (ASSUMED WLOG)

    xd = 1. #+0.5*SPACE_EPS
    # yd = 0. (ASSUMED WLOG)
    # zd = 0. (ASSUMED WLOG)

    sourcePoint = Point(0.,0.,0.)
    detectorPoint = Point(xd,0.,0.)

    detectorNormalVec = np.array([0.,0.,1.])

    targetLineStartLocation = np.array([3.,0.,0.])
    targetLineStopLocation = np.array([3.,5.,5.])

    targetLineNormalVec = np.array([0.,0.,-1.])

    targetLine = LineSegment(targetLineStartLocation, targetLineStopLocation, \
        targetLineNormalVec)


    targetLinePoints = targetLine.getListOfPoints()

    for i, linePoint in enumerate(targetLinePoints):
        time = timeOfLeg(sourcePoint, linePoint) + \
            timeOfLeg(linePoint, detectorPoint)

        leg1 = bounceFactor(sourcePoint.vec, linePoint.vec, targetLineNormalVec)
        leg2 = bounceFactor(linePoint.vec, detectorPoint.vec, detectorNormalVec)

        lightAmount = leg1 * leg2

#        cumulativeAddToTimeLightArray(time, 1)
#        cumulativeAddToTimeLightArray(time, lightAmount)
#        addToTimeLightArray(time, 1)
        addToTimeLightArray(time, lightAmount)

        if i % 10000 == 0:
            print i

#    print TIME_LIGHT_ARRAY


#    plotTimeLightArrayAndApproximationNoBigT([(lineNumPaths(xd, \
#        targetLineStartLocation, targetLineStopLocation), "r-"), \
#        (lineNumPathsNumericDeriv(xd, targetLineStartLocation, \
#        targetLineStopLocation), "g-")], "line_result.png")

#    plotTimeLightArrayAndApproximationNoBigT([(lineCumNumPaths(xd, \
#        targetLineStartLocation, targetLineStopLocation), "r-")], \
#        "line_result.png")

#    plotTimeLightArrayAndApproximationNoBigT([(lineNumPathsNumericDeriv(xd, \
#        targetLineStartLocation, targetLineStopLocation), "r-")], \
#        "line_result.png")

    plotTimeLightArrayAndApproximationNoBigT([(lineResponse(xd, \
        targetLineStartLocation, targetLineStopLocation, targetLineNormalVec, \
        detectorNormalVec), "r-")], "line_result.png")

def doTriangleExperiment():
    # xs = 0. (ASSUMED WLOG)
    # ys = 0. (ASSUMED WLOG)
    # zs = 0. (ASSUMED WLOG)

    xd = 1. #+0.5*SPACE_EPS
    # yd = 0. (ASSUMED WLOG)
    # zd = 0. (ASSUMED WLOG)

    sourcePoint = Point(0.,0.,0.)
    detectorPoint = Point(xd,0.,0.)

    detectorNormalVec = np.array([0.,0.,1.])

    v0 = np.array([1,1,1])
    v1 = np.array([1,2,2])
    v2 = np.array([3,2,3])

    triangle = Triangle(v0, v1, v2)

    trianglePoints = triangle.getListOfPoints()

    for i, trianglePoint in enumerate(trianglePoints):
        time = timeOfLeg(sourcePoint, trianglePoint) + \
            timeOfLeg(trianglePoint, detectorPoint)

        leg1 = bounceFactor(sourcePoint.vec, trianglePoint.vec, triangle.normalVector)
        leg2 = bounceFactor(trianglePoint.vec, detectorPoint.vec, detectorNormalVec)

        lightAmount = leg1 * leg2

#        cumulativeAddToTimeLightArray(time, 1)
#        cumulativeAddToTimeLightArray(time, lightAmount)
#        addToTimeLightArray(time, 1)
        addToTimeLightArray(time, lightAmount)

        if i % 10000 == 0:
            print i

    plotTimeLightArray("triangle_response.png")

def computeL1TLADiff(candidateTLA):
    diffTLA = [abs(i-j)**1.0 for i, j in zip(candidateTLA, TIME_LIGHT_ARRAY)]

    if random.random() < 0.0001:

        p.clf()
        p.plot([i*TIME_STEP for i in range(int(MIN_TIME/TIME_STEP), int(MAX_TIME/TIME_STEP)-1)], diffTLA)
        p.show()

#    minTLA = min(diffTLA)
#    diffTLA.remove(minTLA)

    return sum(diffTLA)

def numericFindSolution(guess, learnRate, tolerance, firstLightFunc, approxFunc, targetWall):
    error = float("Inf")

    numIterations = 0.

    newCurrentParams = guess

    def approxFuncMaker(params, T):
        def simpleApproxFunc(prevT):
            return approxFunc(params, T, prevT)

        return simpleApproxFunc

    while error > tolerance:
        currentParams = newCurrentParams

        candidateTLA = getCandidateTLA(currentParams, targetWall, firstLightFunc, approxFunc)

        error = computeL1TLADiffDropLowest(candidateTLA)

        zsDeviationParams = (currentParams[0] + SPACE_EPS, currentParams[1], currentParams[2])
        xdDeviationParams = (currentParams[0], currentParams[1] + SPACE_EPS, currentParams[2])
        zdDeviationParams = (currentParams[0], currentParams[1], currentParams[2] + SPACE_EPS)

        candidateTLAZS = getCandidateTLA(zsDeviationParams, targetWall, firstLightFunc, approxFunc)
        candidateTLAXD = getCandidateTLA(xdDeviationParams, targetWall, firstLightFunc, approxFunc)
        candidateTLAZD = getCandidateTLA(zdDeviationParams, targetWall, firstLightFunc, approxFunc)

        dErrordZS = computeL1TLADiffDropLowest(candidateTLAZS) - error
        dErrordXD = computeL1TLADiffDropLowest(candidateTLAXD) - error
        dErrordZD = computeL1TLADiffDropLowest(candidateTLAZD) - error

        newCurrentParams = (currentParams[0] - learnRate*dErrordZS,
                            currentParams[1] - learnRate*dErrordXD,
                            currentParams[2] - learnRate*dErrordZD)

        if numIterations % 1000 == 0:
            T = firstLightFunc(currentParams, targetWall)

            plotTimeLightArrayAndApproximation([(approxFuncMaker(currentParams, T), "r-")],
                T, "two_point_plane.png")

        print currentParams, error

        numIterations += 1

    return currentParams

# Assume the two foci are at (-d,0,0) and (d,0,0)
def findSpotOnEggshell(d, T, nu, phi):
    mu = acosh(float(T)/(2*d))

    x = d*cosh(mu)*cos(nu)
    y = d*sinh(mu)*sin(nu)*cos(phi)
    z = d*sinh(mu)*sin(nu)*sin(phi)

    return Point(x,y,z)

# Assume the two foci are at (-d,0,0) and (d,0,0)
def getPlaneTangentTo(d, T, nu, phi):
    a = T/2.
    b = sqrt(T*T/4. - d*d)

    egg = findSpotOnEggshell(d, T, nu, phi)

    planeA = 2.*egg.x/a*a
    planeB = 2.*egg.y/b*b
    planeC = 2.*egg.z/b*b

    return generateWall(np.array([planeA, planeB, planeC]), egg)


#    return plane

def getVectorFromPlaneToPoint(plane, point, pointOnWallFrom1, pointOnWallFrom2):
    zVec = point.distanceToWall(plane)

    pointOnWallFromPoint = plane.nearestPointOnWall(point)

    vecFrom1To2 = np.array([pointOnWallFrom2.x - pointOnWallFrom1.x,
                            pointOnWallFrom2.y - pointOnWallFrom1.y,
                            pointOnWallFrom2.z - pointOnWallFrom1.z])

    vecFrom1ToPoint = np.array([pointOnWallFromPoint.x - pointOnWallFrom1.x,
                            pointOnWallFromPoint.y - pointOnWallFrom1.y,
                            pointOnWallFromPoint.z - pointOnWallFrom1.z])

    xVec = np.dot(vecFrom1To2, vecFrom1ToPoint) / np.linalg.norm(vecFrom1To2)

    if abs(np.linalg.norm(vecFrom1ToPoint)**2 - xVec**2) < DIV0_EPS:
        yVec = 0.
    else:
        yVec = sqrt(np.linalg.norm(vecFrom1ToPoint)**2 - xVec**2)

    return np.array([xVec, yVec, zVec])

# Assume the two foci are at (-d,0,0) and (d,0,0)
def getParams(d, plane):
    p1 = Point(-d, 0, 0)
    p2 = Point(d, 0, 0)

    normalVecPoint = Point(0, 0, SPACE_EPS)
    originVecPoint = Point(0, 0, 0)

    pointOnWallFrom1 = plane.nearestPointOnWall(p1)
    pointOnWallFrom2 = plane.nearestPointOnWall(p2)

    vectorFromPlaneToNormal = getVectorFromPlaneToPoint(plane, normalVecPoint, pointOnWallFrom1,
        pointOnWallFrom2)

    vectorFromPlaneToOrigin = getVectorFromPlaneToPoint(plane, originVecPoint, pointOnWallFrom1,
        pointOnWallFrom2)

    zs = p1.distanceToWall(plane)
    zd = p2.distanceToWall(plane)

    xd = pointOnWallFrom1.distanceToPoint(pointOnWallFrom2)

    vec = (vectorFromPlaneToOrigin - vectorFromPlaneToNormal)/SPACE_EPS

    return (zs, xd, zd), vec

def candidateTLAMaker(d, T):
    def candidateTLAFunc(nuPhiArray):
        nu = nuPhiArray[0]
        phi = nuPhiArray[1]

        plane = getPlaneTangentTo(d, T, nu, phi)
        params, normalVec = getParams(d, plane)

        _, approxFunc = approxFuncMaker()

        candidateTLA, glanceFactors = getCandidateTLA(params, plane, T, normalVec, approxFunc)
        return candidateTLA, glanceFactors

    return candidateTLAFunc


def getErrorMaker(d, T):
    def getError(nuPhiArray):
        candidateTLA = candidateTLAMaker(d, T)(nuPhiArray)
        return computeL1TLADiff(candidateTLA)

    return getError

def gradientDescent(errorFunc, guess, tolerance, learnRate, candidateTLAFunc):
    error = float("inf")

    numIterations = 0.

    newCurrentAngles = guess

    dErrordNu = float("inf")
    dErrordPhi = float("inf")

    while abs(dErrordNu) + abs(dErrordPhi) > tolerance:
        currentAngles = newCurrentAngles
        error = errorFunc(currentAngles)

        nuDeviationArray = np.array([currentAngles[0] + SPACE_EPS, currentAngles[1]])
        phiDeviationArray = np.array([currentAngles[0], currentAngles[1] + SPACE_EPS])

        dErrordNu = errorFunc(nuDeviationArray) - error
        dErrordPhi = errorFunc(phiDeviationArray) - error

        newCurrentAngles = np.array([currentAngles[0] - learnRate*dErrordNu,
                            currentAngles[1] - learnRate*dErrordPhi])

        if numIterations % 10 == 0:

            candidateTLA = candidateTLAFunc(currentAngles)

            plotTimeLightArrayAndCandidate(candidateTLA)

        numIterations += 1

    candidateTLA = candidateTLAFunc(currentAngles)

    plotTimeLightArrayAndCandidate(candidateTLA)

    return currentAngles

#p.show()
#print "hi"

#doTwoPointPlaneExperimentWallDetect(1, 4, pi/2, pi/2)


#    plotTimeLightArrayAndApproximation(
#        [(approxFuncAverageA, "g-"),
#        (approxFuncB, "r-")],
#        T, "two_point_plane.png")

#ls = LineSegment(np.array([1,1,1]), np.array([2,3,4]), 0)
#print [str(i) for i in ls.getListOfPoints()]

#doLineExperiment()

#v1 = np.array([-1,1,1])
#v2 = np.array([-1,-1,1])
#v0 = np.array([3,4,1])

#t = Triangle(v0, v1, v2)

#listOfPoints = t.getListOfPoints()
#print len([str(point) for point in listOfPoints])
#print 2 / (SPACE_EPS*SPACE_EPS)

#for point in listOfPoints:
#    p.plot(point.x, point.y, "bo")

#p.show()

doTriangleExperiment()

#doSpherePointExperimentBothWays()
#doDoubleWallExperiment()

#doTwoPointPlaneExperiment()

#guess = np.array([pi*random.random(), 2*pi*random.random()])

#T = extractTFromTLA()

#print fmin_bfgs(getErrorMaker(sqrt(13)/2, T), guess, disp=True, retall=True, gtol=1e-8)

#print gradientDescent(getErrorMaker(sqrt(13)/2, T), guess, 1e-11, 5e6, candidateTLAMaker(sqrt(13)/2, T))

#targetWall = Wall(0.,0.,1.,0., \
#    np.array([1.,0.,0.]),
#    np.array([0.,1.,0.]))



#numericFindSolution(guess, learnRate, tolerance, firstLightFunc, approxFunc, targetWall)

#TIME_LIGHT_ARRAY = [1.,2.,3.]
#print computeL1TLADiffDropLowest([1.1, 1.8, 3.5])
