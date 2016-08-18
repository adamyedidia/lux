from math import sqrt, pi, sin, cos, cosh, sinh, acosh, asinh, log
import numpy as np
import matplotlib.pyplot as p
import random
from scipy.optimize import fmin_bfgs
import jack3d

TIME_STEP = 0.02

# How many photons were emitted (total) by our laser pulse?
NUM_PHOTONS = 1e10

SEARCH_EPS = 1e-2

TIME_STEP = 0.02
MIN_TIME = 0.
MAX_TIME = 18.

SPACE_EPS = 0.1

# How many photons show up per second
BACKGROUND_NOISE = 1

def poissonify(tla):
    poissonTLA = []
    
    for i in tla:
        poissonTLA.append(np.random.poisson(i))
        
    return poissonTLA
    
def fac(x):
    if x <= 0:
        return 1
    return x*fac(x-1)
    
def logLikelihoodPoisson(l, k):
    if k < 10:
        return k * log(l) - l - log(sqrt(2*pi)) - log(fac(k))    
    else:
        return k * log(l) - l - log(sqrt(2*pi)) - 0.5*log(k) - k*log(k) + k 
        
def scaleWithNumPhotons(model):
    return [i*NUM_PHOTONS for i in model]
    
    
# Poisson distribution is (l^k e^-l)/k!
# Or equivalently (l^k e^-l)/(sqrt(2 pi k)(k/e)^k)

#k log l - l - log(sqrt(2 pi)) - 1/2 log (k) - k log k + k

# Compute log likelihood of tla1 | tla2    
def computeLogLikelihood(tla1, tla2):
    totalLogLikelihood = 0
    
    assert len(tla1) == len(tla2)
    
    for i in range(len(tla1)):
        k = tla1[i]
        l = tla2[i]
        
        totalLogLikelihood += logLikelihoodPoisson(l, k)
        
    return totalLogLikelihood

def addBackgroundNoise(tla):
    backgroundNoisePerTick = BACKGROUND_NOISE * TIME_STEP
        
    return [i + backgroundNoisePerTick for i in tla]

# nu varies between 0 and pi
# phi varies between 0 and pi
# T varies between 2d and inf


def modelMakerTKnown(d, T):
    def modelFunc(nuPhiArray):
        nu = nuPhiArray[0]
        phi = nuPhiArray[1]
                
        assert nu >= 0
        assert nu <= pi
        assert phi >= 0
        assert phi <= pi
        assert T >= 2*d
        
        plane = jack3d.getPlaneTangentTo(d, T, nu, phi)
        params, normalVec = jack3d.getParams(d, plane)
        
        _, approxFunc = jack3d.approxFuncMaker()
        
        model, _ = jack3d.getCandidateTLA(params, plane, T, normalVec, approxFunc)
        
        model = scaleWithNumPhotons(model)
        
        return addBackgroundNoise(model)
        
    return modelFunc

def modelMaker(d):
    def modelFunc(tNuPhiArray):
        T = tNuPhiArray[0]
        nu = tNuPhiArray[1]
        phi = tNuPhiArray[2]
                
        assert nu >= 0
        assert nu <= pi
        assert phi >= 0
        assert phi <= pi
        assert T >= 2*d
        
        plane = jack3d.getPlaneTangentTo(d, T, nu, phi)
        params, normalVec = jack3d.getParams(d, plane)
        
        _, approxFunc = jack3d.approxFuncMaker()
        
        model, _ = jack3d.getCandidateTLA(params, plane, T, normalVec, approxFunc)
        
        model = scaleWithNumPhotons(model)
        
        return addBackgroundNoise(model)
        
    return modelFunc

def getErrorMaker(d, observation):        
    def getError(tNuPhiArray):
        model = modelMaker(d)(tNuPhiArray)
        return -computeLogLikelihood(observation, model)
        
    return getError
    
def getErrorMakerTKnown(d, T, observation):
    def getError(nuPhiArray):
        model = modelMakerTKnown(d, T)(nuPhiArray)
        return -computeLogLikelihood(observation, model)
        
    return getError    

# It's presumed that the bounce wall is at z = 0
# It's also presumed that the two foci of the ellipsoid are at 
# (-d,0,0) and (d,0,0)
def getObservationTwoPointPlane(d, tNuPhiArray):
    model = modelMaker(d)(tNuPhiArray)
        
    observation = poissonify(model)
    
    return observation
    
def getObservationTwoPointPlaneTKnown(d, T, nuPhiArray):
    model = modelMakerTKnown(d, T)(nuPhiArray)
        
    observation = poissonify(model)
    
    return observation    

def incrIndex(currentVec, index):
    return [val + SEARCH_EPS*(index == i) for i, val in enumerate(currentVec)]

def exhaustiveSearchHack(errorFunc, minNu, minPhi, maxNu, maxPhi, observation):
    
    bestError = float("Inf")
    bestNuPhiArray = None
    
    nu = minNu
    
    while nu < maxNu:
        phi = minPhi
        
        while phi < maxPhi:
            
            nuPhiArray = np.array([nu, phi])
            
            print (nu, phi)
            
            error = errorFunc(nuPhiArray)
            
            if error < bestError:
                bestError = error
                bestMinNuArray = nuPhiArray
                
                
            phi += SEARCH_EPS
        
        nu += SEARCH_EPS
        
    return bestMinNuArray

def exhaustiveSearch(errorFunc, min, max, observation):
    assert len(minVec) == len(maxVec)
    
    dim = len(minVec)

    currentVec = minVec

    while True:
        currentIndex = 0
        
        while currentIndex:
        
            newCurrentVec = incrIndex(currentVec, currentIndex)
        
            if newCurrentVec[currentIndex]:
                pass
        

def gradientDescent(errorFunc, guess, tolerance, learnRate, modelFunc, observation):
    error = float("Inf")
    
    numIterations = 0.
    
    newCurrentTNuPhi = guess
    
    dim = len(guess)
    
    gradVec = np.array([float("inf")]*dim)
    
    while np.linalg.norm(gradVec) > tolerance:    
        currentTNuPhi = newCurrentTNuPhi
        error = errorFunc(currentTNuPhi)
        
        arrayOfDeviationArrays = [np.array([val+SPACE_EPS*(i==j) for i, val in enumerate(currentTNuPhi)])
                                for j in range(dim)]
        
#        nuDeviationArray = np.array([currentTNuPhi[0] + SPACE_EPS, currentTNuPhi[1], currentTNuPhi[2]])
#        phiDeviationArray = np.array([currentTNuPhi[0], currentTNuPhi[1] + SPACE_EPS, currentTNuPhi[2]])
#        tDeviationArray = np.array([currentTNuPhi[0], currentTNuPhi[1], currentTNuPhi[2] + SPACE_EPS])
 
        gradVec = np.array([errorFunc(deviationArray) - error for deviationArray in arrayOfDeviationArrays])
        
#        dErrordT = errorFunc(tDeviationArray) - error
#        dErrordNu = errorFunc(nuDeviationArray) - error
#        dErrordPhi = errorFunc(phiDeviationArray) - error
                
        newCurrentTNuPhi = currentTNuPhi - learnRate*gradVec        
                
#        newCurrentTNuPhi = np.array([currentTNuPhi[0] - learnRate*dErrordNu,
#                            currentTNuPhi[1] - learnRate*dErrordPhi,
#                            currentTNuPhi[2] - learnRate*dErrordT])
#                            
        if numIterations % 100 == 0:      
            model = modelFunc(currentTNuPhi)               
            jack3d.plotArray(model)
            jack3d.plotArray(observation)  
            p.savefig("iter_" + str(numIterations) + ".png")            
            
        print currentTNuPhi, error    
            
        numIterations += 1
        
    model = modelFunc(currentTNuPhi)
        
    jack3d.plotTimeLightArrayAndCandidate(candidateTLA)    
        
    return currentTNuPhi                

        
d = 1
T = 4
nu = pi/4
phi = pi/4

modelFunc = modelMakerTKnown(d, T)

model = modelFunc(np.array([nu, phi]))

observation = getObservationTwoPointPlaneTKnown(d, T, np.array([nu, phi]))

jack3d.plotArray(model)
jack3d.plotArray(observation)

#print model

errorFunc = getErrorMakerTKnown(d, T, observation)

print errorFunc(np.array([nu, phi]))
print errorFunc(np.array([pi/2, pi/2]))
#print computeLogLikelihood(observation, wrongModel)

p.show()

#gradientDescent(errorFunc, np.array([pi/2, pi/2]), 1e-8, 1e-5, modelFunc, observation)

print exhaustiveSearchHack(errorFunc, 0, 0, pi/2, pi/2, observation)