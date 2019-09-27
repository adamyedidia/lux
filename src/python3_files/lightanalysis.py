from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as p
from math import log, sqrt, sin
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import sys

REQUEST = sys.argv[1]
COLORMAP = sys.argv[2]
EPS = 0.05
STEP = 0.1
MAX_XT = 10.
MAX_YT = 10.
# I think STEP**2 had better be more than EPS or there may be problems

XS = 0.
YS = 3.
XB = 3.
YB = 0.

TIME_STEP = 0.1
MIN_TIME = 0.
MAX_TIME = 100.

X_WALL_STEP = 0.01
MIN_X_WALL = XS - 5.
MAX_X_WALL = MAX_XT + 5.

TIME_LIGHT_ARRAY = [0.] * int((MAX_TIME - MIN_TIME) / TIME_STEP)

def lightFromTrip(deltaX, deltaY):
    result = EPS/sqrt(deltaX**2 + deltaY**2)
#    result = EPS/(deltaX**2 + deltaY**2)
    
    if result > 0.5:
        print("Warning: large trip value", result, "deltaX", deltaX, "deltaY", deltaY)
    
    return min(result, 0.5)

def timeOfTrip(deltaX, deltaY):
    return sqrt(deltaX**2 + deltaY**2)

def lightFromGreatJourney(x, xs, ys, xb, yb, xt, yt):
    firstTrip = 1
    secondTrip = lightFromTrip(xt-xb, yt-yb)
    thirdTrip = lightFromTrip(x-xt, yt-yb)
    fourthTrip = lightFromTrip(x-xs, ys-yb)

    return firstTrip*secondTrip*thirdTrip*fourthTrip

def timeOfGreatJourney(x, xs, ys, xb, yb, xt, yt):
    firstTrip = timeOfTrip(xs-xb, ys-yb)
    secondTrip = timeOfTrip(xb-xt, yb-yt)
    thirdTrip = timeOfTrip(xt-x, yt-yb)
    fourthTrip = timeOfTrip(x-xs, yb-ys)
    
    return firstTrip+secondTrip+thirdTrip+fourthTrip

def addToTimeLightArray(time, lightAmount):
    timeIndex = int((time-MIN_TIME)/TIME_STEP)
    
    if timeIndex < len(TIME_LIGHT_ARRAY):
        TIME_LIGHT_ARRAY[timeIndex] += lightAmount
        
def accountForPointInTLA(xs, ys, xb, yb, xt, yt):   
    for xScaledUp in range(int(MIN_X_WALL/X_WALL_STEP), int(MAX_X_WALL/X_WALL_STEP)):
        x = xScaledUp * X_WALL_STEP
        
        t = timeOfGreatJourney(x, xs, ys, xb, yb, xt, yt)
        l = lightFromGreatJourney(x, xs, ys, xb, yb, xt, yt)
        
        addToTimeLightArray(t, l)
#        addToTimeLightArray(t, 1)
        
def accountForSurfaceInTLA(xs, ys, xb, yb, surfaceFunc, minXT, maxXT, xtStep):
    for xtScaledUp in range(int(minXT / xtStep), int(maxXT / xtStep)):
        xt = xtScaledUp * xtStep
        yt = surfaceFunc(xt)
        
        accountForPointInTLA(xs, ys, xb, yb, xt, yt)
        
def plotTimeLightArray():
    p.clf()
    p.plot([i*TIME_STEP for i in range(int(MIN_TIME/TIME_STEP), int(MAX_TIME/TIME_STEP))], TIME_LIGHT_ARRAY)
    p.savefig("tla.png")
    p.show()
    
def computeTotalLight(xs, ys, xb, yb):
    coordTuples = []

    for xtScaledUp in range(1, int(MAX_XT/STEP)):
        xt = xtScaledUp * STEP
    
        for ytScaledUp in range(1, int(MAX_YT/STEP)):
            yt = ytScaledUp * STEP
        
            lightComeBack = quad(lightFromGreatJourney, -np.inf, np.inf, args=(xs, ys, xb, yb, xt, yt))[0]
        
            coordTuples.append((xt, yt, log(lightComeBack)))

    return coordTuples
        
def computeFirstLight(xs, ys, xb, yb):
    coordTuples = []
    
    for xtScaledUp in range(1, int(MAX_XT/STEP)):
        xt = xtScaledUp * STEP
    
        for ytScaledUp in range(1, int(MAX_YT/STEP)):
            yt = ytScaledUp * STEP
            
            k = (ys - yb) / float(yt - 2*yb + ys)
            
            x = (1 - k) * xs + k * xt
        
            lightComeBack = lightFromGreatJourney(x, xs, ys, xb, yb, xt, yt)
        
            coordTuples.append((xt, yt, log(lightComeBack)))
            
    return coordTuples
 
 
def computeTimeOfFirstLight(xs, ys, xb, yb):    
    coordTuples = []
    
    for xtScaledUp in range(1, int(MAX_XT/STEP)):
        xt = xtScaledUp * STEP
    
        for ytScaledUp in range(1, int(MAX_YT/STEP)):
            yt = ytScaledUp * STEP    
    
            k = (ys - yb) / float(yt - 2*yb + ys)
            
            x = (1 - k) * xs + k * xt
            
            timeComeBack = timeOfGreatJourney(x, xs, ys, xb, yb, xt, yt)
            
            coordTuples.append((xt, yt, log(timeComeBack)))
            
    return coordTuples

def makeFirstLightPlot(xs, ys, xb, yb, xt, yt):
    p.clf()
    p.plot(xs, ys, "ko")
    p.plot(xb, yb, "wo")        
    p.plot(xt, yt, "go")
    
    p.plot([xs, xb], [ys, yb], "k-")
    p.plot([xb, xt], [yb, yt], "k-")
    
    k = (ys - yb) / float(yt - 2*yb + ys)
    
    x = (1 - k) * xs + k * xt
    
    p.plot([xt, x], [yt, yb], "r-")
    p.plot([x, xs], [yb, ys], "r-")
    
    p.savefig("first_light_path.png")
    p.show()
    
    
def makeColorPlot(coordTuples, xs, ys, xb, yb):        
    vmin = min([i[2] for i in coordTuples])
    vmax = max([i[2] for i in coordTuples])
        
    norm = Normalize(vmin, vmax)
    prism = cm.get_cmap(name=COLORMAP)
    scalarMappable = cm.ScalarMappable(norm=norm, cmap=prism)

    p.plot(xs, ys, "ko")
    p.plot(xb, yb, "wo")

    for t in coordTuples:
#        print t[0], t[1], t[2]
        p.plot(t[0], t[1], color=scalarMappable.to_rgba(t[2]), marker=".")
        
    p.savefig("light_diagram_" + REQUEST + "_" + COLORMAP + ".png")
#    p.show()
    
def colorMain(): 
    if REQUEST == "total":
        coordTuples = computeTotalLight(XS, YS, XB, YB)
    elif REQUEST == "first":
        coordTuples = computeFirstLight(XS, YS, XB, YB)
    elif REQUEST == "first_time":
        coordTuples = computeTimeOfFirstLight(XS, YS, XB, YB)
    
    print(coordTuples)    
    makeColorPlot(coordTuples, XS, YS, XB, YB)
    
def tlaMain():
#    accountForPointInTLA(0, 3, 3, 0, 6, 4)

#    surfaceFunc = lambda x : -2./3.*x + 22./3. 
#    surfaceFunc = lambda x : -x + 10.
    surfaceFunc = lambda x : sin(x) + 5
    
    accountForSurfaceInTLA(0., 3., 3., 0., surfaceFunc, 2., 8., 0.01)
        
    plotTimeLightArray()
    print(sum(TIME_LIGHT_ARRAY))
    
if __name__ == "__main__":
    tlaMain()    