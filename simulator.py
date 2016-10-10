from jack3d import *
import numpy as np

def reflectThroughWall(listOfPoints, wall):
    return [wall.reflectThroughWall(point) for point in listOfPoints]

def multipleBounceProblemWallPointDetector(listOfWalls, sourcePoint, finalPoint, finalWall, squareRadius=2.):
    # This is a general function for simulating problems of multiple bounces
    # The detector is presumed to be a point on a wall

    sourcePoint.personalTLA[0] += 1.0

    effectiveSourcePoint = sourcePoint

    listOfListsOfPointsInWalls = [wall.getListOfPoints(squareRadius, \
        wall.getRandomExampleVector()) for wall in listOfWalls]

    sourcePointStrength = 1.
    listOfWallStrengths = [1. for wall in listOfWalls]

    for i, wall2 in enumerate(listOfWalls):
        print i

        receivingWall = listOfWalls[i]

        if i > 0:
            # reflect all walls through the second-to-latest wall
            specularWall = listOfWalls[i-1]

            effectiveSourcePoint = listOfWalls[i-1].reflectThroughWall(effectiveSourcePoint)

            sourcePointStrength = sourcePointStrength*specularWall.specularReflectance

            for j, wall1 in enumerate(listOfWalls[:i-1]):
                listOfListsOfPointsInWalls[j] = \
                    reflectThroughWall(listOfListsOfPointsInWalls[j], listOfWalls[i-1])

                listOfWallStrengths[j] *= specularWall.specularReflectance

        transferLightFromOneListOfPointsToAnother([effectiveSourcePoint], \
            listOfListsOfPointsInWalls[i], receivingWall, sourcePointStrength)

        for j, wall1 in enumerate(listOfWalls[:i]):
            transferLightFromOneListOfPointsToAnother(listOfListsOfPointsInWalls[j], \
                listOfListsOfPointsInWalls[i], receivingWall, \
                wall1.lambertianReflectance*listOfWallStrengths[j])

    # Now do the thing for the final wall!
    specularWall = listOfWalls[-1]

    listOfPointsInFinalWall = finalWall.getListOfPoints(squareRadius, \
        wall.getRandomExampleVector())

    for j, wall1 in enumerate(listOfWalls[:-1]):
        listOfListsOfPointsInWalls[j] = \
            reflectThroughWall(listOfListsOfPointsInWalls[j], listOfWalls[-1])

        listOfWallStrengths[j] *= specularWall.specularReflectance

    transferLightFromOneListOfPointsToAnother([effectiveSourcePoint], \
        listOfPointsInFinalWall, finalWall, sourcePointStrength)

    for j, wall1 in enumerate(listOfWalls[:i]):
        transferLightFromOneListOfPointsToAnother(listOfListsOfPointsInWalls[j], \
            listOfPointsInFinalWall, finalWall, \
            wall1.lambertianReflectance*listOfWallStrengths[j])

def transferLightFromOneListOfPointsToAnother(list1, list2, wall2, reflectanceFactor):
    if reflectanceFactor == 0.0:
        return

    print list1[0].personalTLA

    for point1 in list1:
        for point2 in list2:
            t = timeOfLeg(point1, point2)
            intensityFactor = lightFactorWallToWall(point1, point2, wall2) * reflectanceFactor

            for tlaIndex, lightAmount in enumerate(point1.personalTLA):
                if tlaIndex == 0 and len(list1) == 1:
                    print tlaIndex, lightAmount, t, intensityFactor

                addToTimeLightArrayObject(convertTLAIndexToTime(tlaIndex)+t, lightAmount*intensityFactor, \
                    point2.personalTLA)

    print list2[0].personalTLA

sourcePoint = Point(0,0,0)

# z = 0
reflectorWall = Wall(0,0,1,0, np.array([1,0,0]), np.array([0,1,0]), lambertianReflectance=0.5, \
    specularReflectance=0.01)

# x + y + z = 1
targetWall = Wall(1,1,1,1, np.array([-2,1,1]), np.array([-2,1,1]))

# z = 1
detectorWall = Wall(0,0,1,1, np.array([1,0,0]), np.array([0,1,0]))

detectorPoint = Point(-1,-1,1)

multipleBounceProblemWallPointDetector([reflectorWall, targetWall, detectorWall],
                                        sourcePoint, detectorPoint, detectorWall)

plotArray(detectorPoint.personalTLA)
p.savefig("part_specular.png")
p.show()
