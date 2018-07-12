from __future__ import division
import numpy as np
import matplotlib.pyplot as p
from matplotlib.colors import LinearSegmentedColormap
from math import log, pi, exp, sqrt, sin, cos
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator
from matplotlib import rcParams
from pynverse import inversefunc

def createListOfTuplesForSingleCycle(currentVal, cycleLength):
    if currentVal > 0.5:
        minBrightness = currentVal*2-1
        maxBrightness = 1
    
    else:
        minBrightness = 0
        maxBrightness = (currentVal + cycleLength/2)*2
        
    firstStage = (currentVal, minBrightness, minBrightness)
    secondStage = (currentVal + cycleLength/2, maxBrightness, maxBrightness)

    return [firstStage, secondStage]

def createTupleOfTuplesForSingleColor(numCycles, color):
    cycleLength = 1/numCycles
           
        
    if color == "red":
        currentVal = 0
        listOfTuples = []
    
    if color == "blue":
        currentVal = cycleLength/3
        listOfTuples = [(0,0,0)]
    
    if color == "green":
        currentVal = 2*cycleLength/3
        listOfTuples = [(0,0,0), (cycleLength/6, cycleLength/3, cycleLength/3)]
    
    while currentVal + cycleLength <= 1:
        listOfTuples.extend(createListOfTuplesForSingleCycle(currentVal, cycleLength))
        currentVal += cycleLength
        
    if color == "red":
        listOfTuples.append((1,1,1))    
        
    if color == "blue":
        listOfTuples.append((currentVal, currentVal*2-1, currentVal*2-1))
        listOfTuples.append((currentVal + cycleLength/2, 1, 1))
        listOfTuples.append((1,1,1))
        
    if color == "green":
        listOfTuples.append((currentVal, currentVal*2-1, currentVal*2-1))
        listOfTuples.append((1, 1, 1))
        
    return tuple(listOfTuples)
    
def createCMapDictHelix(numCycles):
    returnDict = {}
    
    for color in ["red", "blue", "green"]:
        returnDict[color] = createTupleOfTuplesForSingleColor(numCycles, color)
#        print color, returnDict[color]
        
        
    return returnDict

cdict5 = createCMapDictHelix(10)

helix = LinearSegmentedColormap("helix", cdict5)

p.register_cmap(cmap=helix)

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def logLaplace(x, mu, sigma):
    return exp(-abs(log(x)-mu)/(sigma))/(2*x*sigma)

def cumLogLaplace(y, mu, sigma):
    if y <= 0:
        return 0
    else:
        return (1 + sign(log(y)-mu)*(1-exp(-abs(log(y)-mu)/sigma)))/2

def cumLogLaplaceMaker(mu, sigma):
    def cumLogLaplaceParametrized(y):
        if y <= 0:
            return 0
        else:
            return (1 + sign(log(y)-mu)*(1-exp(-abs(log(y)-mu)/sigma)))/2
    return cumLogLaplaceParametrized


xRange = np.linspace(0.01, 4, 200)
yRange = np.linspace(0.01, 0.99, 50)

mu = 0
sigma = 0.2

cumLogLaplaceParametrized = cumLogLaplaceMaker(mu, sigma)
inverseCumLogLaplaceParametrized = inversefunc(cumLogLaplaceParametrized, domain=[0.001, 100])

cumLogLaplaceVectorized = np.vectorize(cumLogLaplaceParametrized)
inverseCumLogLaplaceVectorized = np.vectorize(inverseCumLogLaplaceParametrized)

p.plot(xRange, [logLaplace(x, mu, sigma) for x in xRange])
p.plot(xRange, [cumLogLaplaceParametrized(x) for x in xRange])
p.plot(yRange, [inverseCumLogLaplaceParametrized(x) for x in yRange])
p.show()

class LogLaplaceScale(mscale.ScaleBase):
    name = 'log_laplace'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)

        self.mu = kwargs["mu"]
        self.sigma = kwargs["sigma"]

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The MercatorLatitudeTransform class is defined below as a
        nested class of this one.
        """
        return self.LogLaplaceTransform(self.mu, self.sigma)

    class LogLaplaceTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, mu, sigma):
            mtransforms.Transform.__init__(self)
            self.mu = mu
            self.sigma = sigma

        def transform_non_affine(self, a):
#            return np.power(a, 0.5)
            cumLogLaplaceVectorized = np.vectorize(cumLogLaplaceMaker(self.mu, self.sigma))
            return cumLogLaplaceVectorized(a)

        def inverted(self):
            return LogLaplaceScale.InvertedLogLaplaceTransform(self.mu, self.sigma)

    def set_default_locators_and_formatters(self, axis):
        pass

    class InvertedLogLaplaceTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)
            self.mu = mu
            self.sigma = sigma

        def transform_non_affine(self, a):
            
            cumLogLaplaceVectorized = np.vectorize(inversefunc(cumLogLaplaceMaker(self.mu, \
                self.sigma), domain=[1e-5, float("Inf")]))
            
            return inverseCumLogLaplaceVectorized(a)

        def inverted(self):
            return LogLaplaceScale.LogLaplaceTransform(self.mu, self.sigma)

mscale.register_scale(LogLaplaceScale)

p.plot(xRange, xRange)
ax = p.gca()
ax.set_ylim(0.01, 10)
ax.set_xlim(0.01, 10)
ax.set_yscale("log_laplace", mu=0, sigma=0.04)

p.yticks([0.5, 0.9, 0.99, 1.01, 1.1, 2])

p.show()


#def cumLogLaplace(x, mu, sigma):
#    return 

