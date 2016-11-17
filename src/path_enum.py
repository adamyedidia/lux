from math import pi, sqrt, acos, floor, cos
import numpy as np

def distance(p1, p2):
    return sqrt((p1.position[0] - p2.position[0])**2 + \
            (p1.position[1] - p2.position[1])**2 + \
            (p1.position[2] - p2.position[2])**2)

def angleBetweenUnitVectors(vec1, vec2):
    return acos(np.dot(vec1, vec2))

def normalize(vec):
    return vec/np.linalg.norm(vec)

def vectorFromPatchToPatch(p1, p2):
    return p2.position - p1.position

def legFactorLambertian(p1, p2):

    p1ToP2Vector = vectorFromPatchToPatch(p1, p2)

    p1ToP2Angle = angleBetweenUnitVectors(normalize(vectorFromPatchToPatch(p1, \
        p2)), p2.normal)

    return cos(p1ToP2Angle)*p2.area/(2*pi*np.linalg.norm(p1ToP2Vector)**2)

# Assume directional source and directional detector
def pathEnum(object, sourcePatch, sourceReflectorPatch, detectorReflectorPatch, \
    detectorPatch, reflectorNormal, detectorNormal, deltaT, maxT):

    timeHistogram = [0] * (maxT / deltaT)

    sourceToSourceReflectorDistance = distance(sourcePatch, sourceReflector)
    detectorReflectorToDetectorDistance = distance(detectorPatch, detector)

    sourceToSourceReflectorAngle = \
        angleBetweenUnitVectors(normalize(vectorFromPatchToPatch(sourcePatch, \
            sourceReflectorPatch), reflectorNormal))


    sourceToSourceReflectorFactor = cos(sourceToSourceReflectorAngle)
    detectorReflectorToDetectorFactor = 1 # Assume detector is head-on

    for patch in object.patches:
        t = sourceToSourceReflectorDistance + \
            distance(sourceReflectorPatch, patch) + \
            distance(patch, detectorReflectorPatch) + \
            detectorReflectorToDetectorDistance

        intensity = sourceToSourceReflectorFactor * \
            legFactorLambertian(sourceReflectorPatch, patch) * \
            legFactorLambertian(patch, detectorReflectorPatch) * \
            detectorReflectorToDetectorFactor

        if t < maxT:
            histogramIndex = floor(t / deltaT)
            timeHistogram[histogramIndex] += intensity

    return timeHistogram
