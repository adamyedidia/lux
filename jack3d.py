from math import sqrt, pi
import numpy as np
import matplotlib.pyplot as p
import random

TIME_STEP = 0.1
MIN_TIME =0.
MAX_TIME =18.

SPACE_EPS = 0.05

TIME_LIGHT_ARRAY = [0.] * int(MAX_TIME/TIME_STEP - MIN_TIME/TIME_STEP)

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
        self.vec = np.array([x,y,z])

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"

    def distanceToPoint(self, otherPoint):
        return sqrt((self.x-otherPoint.x)*(self.x-otherPoint.x) \
                    +(self.y-otherPoint.y)*(self.y-otherPoint.y) \
                    +(self.z-otherPoint.z)*(self.z-otherPoint.z))

    def distanceToWall(self, wall):
        return abs(self.signedDistanceToWall(wall))
                
    def signedDistanceToWall(self, wall):
        return -(self.x*wall.a + self.y*wall.b + self.z*wall.c + wall.d) / \
                sqrt(wall.a*wall.a + wall.b*wall.b + wall.c*wall.c)

class Wall:
    def __init__(self, a, b, c, d, unitVector1, unitVector2):
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
           
    def __str__(self):
        return str(self.a) + "x + " + \
            str(self.b) + "y + " + \
            str(self.c) + "z = " + \
            str(self.d)       
           
    def getListOfPoints(self, squareRadius, exampleVector):
        listOfPoints = []
        
        adjustedSquareRadius = int(squareRadius/SPACE_EPS)
        
        for uvFactor1Large in range(-adjustedSquareRadius, adjustedSquareRadius+1):
            uvFactor1 = uvFactor1Large*SPACE_EPS
            
            for uvFactor2Large in range(-adjustedSquareRadius, adjustedSquareRadius+1):
                uvFactor2 = uvFactor2Large*SPACE_EPS
                
                resultVector = uvFactor1*self.unitVector1 + \
                    uvFactor2*self.unitVector2 + \
                    exampleVector
                
                listOfPoints.append(Point(resultVector[0], resultVector[1], 
                    resultVector[2]))
                                        
        return listOfPoints
               
    def reflectThroughWall(self, point):
        signedDistanceToWall = point.signedDistanceToWall(self)
        
        newPointVec = point.vec + 2 * self.normalVector * signedDistanceToWall
        
        return Point(newPointVec[0], newPointVec[1], newPointVec[2])
               
def addToTimeLightArray(time, lightAmount):
    timeIndex = int((time-MIN_TIME)/TIME_STEP)
    
    if timeIndex < len(TIME_LIGHT_ARRAY) and timeIndex > 0:
        TIME_LIGHT_ARRAY[timeIndex] += lightAmount               
    
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
        
def doSpherePointExperiment():
    T=1
        
    sourcePoint = Point(0,0,T)
        
    # Wall at z=0    
    wall = Wall(0,0,1.,0, \
        np.array([1.,0,0]), \
        np.array([0,1.,0]))
            
    wallPoints = wall.getListOfPoints(2., np.array([0,0,0]))
    
    for wallPoint in wallPoints:    
        time = timeOfLeg(sourcePoint, wallPoint)
        lightAmount = lightFactorPointToWall(sourcePoint, wallPoint, wall)
                
#        print (time,lightAmount), wallPoint
        
        addToTimeLightArray(time, lightAmount)
#        addToTimeLightArray(time, 1)
    
    def approxFunc(t):  
        if t < 0.:
            return 0.
                
        return T*TIME_STEP/(2*(T+t)*(T+t))    
        
    plotTimeLightArrayAndApproximation(approxFunc, T, "spherepoint.png")
        
    
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
        
        print i
            
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
            
def doTwoPointPlaneExperiment():
    # xs = 0. (ASSUMED WLOG)
    # ys = 0. (ASSUMED WLOG)
    zs = 1. #+0.5*SPACE_EPS    
    
    xd = 3. #+0.5*SPACE_EPS    
    # yd = 0. (ASSUMED WLOG)
    zd = 3. #+0.5*SPACE_EPS    
    
    x = xd*(zs/(zd+zs))
    
    sourcePoint = Point(0.,0.,zs)
    detectorPoint = Point(xd,0.,zd)
    
    # We know the location of the target because we did a knowledge inversion
    targetWall = Wall(0.,0.,1.,0., \
        np.array([1.,0.,0.]),
        np.array([0.,1.,0.]))
        
    T = timeOfLeg(sourcePoint, targetWall.reflectThroughWall(detectorPoint))
#    print T    
        
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

def funcMaker():
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
            
    def pathStrengthCenterSourceToWall(params, xb, bOfT):
        zs, xd, zd = params
        
        return zs*SPACE_EPS*SPACE_EPS/ \
            (4*pi*(xb*xb + zs*zs)**1.5)        
            
    def pathStrengthCenterWallToDetector(params, xb, bOfT):
        zs, xd, zd = params
        
        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-xd)**2 + \
            zd*zd))             
            
    def pathStrengthAMinusSourceToWall(params, xb, aOfT):
        zs, xd, zd = params
        
        return zs*SPACE_EPS*SPACE_EPS / \
            (4*pi*((xb-aOfT)**2 + zs*zs)**1.5)
            
    def pathStrengthAMinusWallToDetector(params, xb, aOfT):
        zs, xd, zd = params
        
        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-aOfT-xd)**2 + \
            zd*zd))
            
    def pathStrengthAPlusSourceToWall(params, xb, aOfT):
        zs, xd, zd = params
        
        return zs*SPACE_EPS*SPACE_EPS / \
            (4*pi*((xb+aOfT)**2 + zs*zs)**1.5)
            
    def pathStrengthAPlusWallToDetector(params, xb, aOfT):
        zs, xd, zd = params
        
        return SPACE_EPS*SPACE_EPS / (2*pi*((xb-aOfT-xd)**2 + \
            zd*zd))            
            
    def pathStrengthCenter(params, xb, bOfT):
        return pathStrengthCenterSourceToWall(params, xb, bOfT) * \
                pathStrengthCenterWallToDetector(params, xb, bOfT)           
            
    def pathStrengthB(params, xb, bOfT):
        return pathStrengthBSourceToWall(params, xb, bOfT) * \
                pathStrengthBWallToDetector(params, xb, bOfT)
                
    def pathStrengthAMinus(params, xb, aOfT):
        return pathStrengthAMinusSourceToWall(params, xb, aOfT) * \
                pathStrengthAMinusWallToDetector(params, xb, aOfT)
                
    def pathStrengthAPlus(params, xb, aOfT):
        return pathStrengthAPlusSourceToWall(params, xb, aOfT) * \
                pathStrengthAPlusWallToDetector(params, xb, aOfT)
        
    def pathStrengthWeightAverage(params, xb, aOfT, bOfT):
        return 0.5*pathStrengthB(params, xb, bOfT) + \
                0.25*(pathStrengthAMinus(params, xb, aOfT) + \
                    pathStrengthAPlus(params, xb, aOfT))
                    
    def approxFunc(params, T, prevT):
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
        
        return numPaths(aOfT, bOfT, dadt, dbdt)*pathStrengthWeightAverage(params, xb, aOfT, bOfT)        
        
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

#def getFirstNonzeroIndexFromTLA():
#    for i, val in enumerate(TIME_LIGHT_ARRAY):
#        if val != 0.:
#            return i

#def getValsFromTLATwoPointPlane():
#    nonzeroIndex = getFirstNonzeroIndexFromTLA()
#    
#    T = MIN_TIME + (nonzeroIndex+0.5)*TIME_STEP
#
#    funcOfTPlus12 = TIME_LIGHT_ARRAY[nonzeroIndex + 11]
#    funcOfTPlus15 = TIME_LIGHT_ARRAY[nonzeroIndex + 14]
#    
#    return (T, funcOfTPlus12, funcOfTPlus15-funcOfTPlus12)
    
    
# Assume MIN_TIME = 0    
def getCandidateTLA(params, targetWall, firstLightFunc, approxFunc):
    T = firstLightFunc(params, targetWall)
    
    tla = [0] * int(T/TIME_STEP)
    
    tla += [approxFunc(params, T, i+0.5) for i in range(int((MAX_TIME-T)/TIME_STEP))]
    
    return tla
        
def computeL1TLADiffDropLowest(candidateTLA):
    diffTLA = [abs(i-j)**0.8 for i, j in zip(candidateTLA, TIME_LIGHT_ARRAY)]
    
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
    
#    plotTimeLightArrayAndApproximation(
#        [(approxFuncAverageA, "g-"),
#        (approxFuncB, "r-")],
#        T, "two_point_plane.png")

#doSpherePointExperiment()        
#doDoubleWallExperiment()  
  
doTwoPointPlaneExperiment()
firstLightFunc, approxFunc = funcMaker()

guess = (1.2,2.8,3.2)
learnRate = 100
tolerance = 1e-6

targetWall = Wall(0.,0.,1.,0., \
    np.array([1.,0.,0.]),
    np.array([0.,1.,0.]))

numericFindSolution(guess, learnRate, tolerance, firstLightFunc, approxFunc, targetWall)