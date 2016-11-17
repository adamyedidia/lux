from jack3d import *
import numpy as np

NUM_PHOTONS = 1e10

def reflectThroughWall(listOfPoints, wall):
    return [wall.reflectThroughWall(point) for point in listOfPoints]

def multipleBounceProblemWallPointDetector(listOfWalls, listOfExamplePoints, sourcePoint, \
    finalPoint, finalWall, squareRadius=3.):
    # This is a general function for simulating problems of multiple bounces
    # The detector is presumed to be a point on a wall

    sourcePoint.personalTLA[0] += NUM_PHOTONS

    effectiveSourcePoint = sourcePoint

    listOfListsOfPointsInWalls = [wall.getListOfPoints(squareRadius, \
        examplePoint) for wall, examplePoint in zip(listOfWalls, listOfExamplePoints)]

#    print [[str(p) for p in l] for l in listOfListsOfPointsInWalls]

    sourcePointStrength = 1.
    listOfWallStrengths = [1. for wall in listOfWalls]

    for i, wall2 in enumerate(listOfWalls):
#        print i

        receivingWall = listOfWalls[i]

        if i > 0:
            # reflect all walls through the second-to-latest wall
            specularWall = listOfWalls[i-1]

            print "Reflecting source point", effectiveSourcePoint, "through", \
                listOfWalls[i-1]
            effectiveSourcePoint = listOfWalls[i-1].reflectThroughWall(effectiveSourcePoint)

    #        print "strength", sourcePointStrength, specularWall.specularReflectance, specularWall
            sourcePointStrength *= specularWall.specularReflectance
    #        print "strength", sourcePointStrength, specularWall.specularReflectance, specularWall


            for j, wall1 in enumerate(listOfWalls[:i-1]):
                print "Reflecting wall", wall1, "through wall", listOfWalls[i-1]

                listOfListsOfPointsInWalls[j] = \
                    reflectThroughWall(listOfListsOfPointsInWalls[j], listOfWalls[i-1])

                listOfWallStrengths[j] *= specularWall.specularReflectance

        transferLightFromOneListOfPointsToAnother([effectiveSourcePoint], \
            listOfListsOfPointsInWalls[i], receivingWall, sourcePointStrength)

        for j, wall1 in enumerate(listOfWalls[:i]):
            print "Transfering light from", wall1, "to", wall2

            transferLightFromOneListOfPointsToAnother(listOfListsOfPointsInWalls[j], \
                listOfListsOfPointsInWalls[i], receivingWall, \
                wall1.lambertianReflectance*listOfWallStrengths[j])

    # Now do the thing for the final wall!
    specularWall = listOfWalls[-1]

#    listOfPointsInFinalWall = finalWall.getListOfPoints(squareRadius, \
#        wall.getRandomExampleVector())

# I don't know why I ever had the above commented-out line --Future Adam

    for j, wall1 in enumerate(listOfWalls[:-1]):
        listOfListsOfPointsInWalls[j] = \
            reflectThroughWall(listOfListsOfPointsInWalls[j], listOfWalls[-1])

        listOfWallStrengths[j] *= specularWall.specularReflectance

    print effectiveSourcePoint, sourcePoint
    transferLightFromOneListOfPointsToAnother([effectiveSourcePoint], \
        [finalPoint], finalWall, sourcePointStrength)

    for j, wall1 in enumerate(listOfWalls[:i]):
        transferLightFromOneListOfPointsToAnother(listOfListsOfPointsInWalls[j], \
            [finalPoint], finalWall, \
            wall1.lambertianReflectance*listOfWallStrengths[j])

def transferLightFromOneListOfPointsToAnother(list1, list2, wall2, reflectanceFactor):
    if reflectanceFactor == 0.0:
        return


#    print [str(p) for p in list1]
#    print [str(p) for p in list2]



    for i, point1 in enumerate(list1):
        if i % 100 == 0:
            print i, "/", len(list1)
        for point2 in list2:
            t = timeOfLeg(point1, point2)
#            print point1, point2, wall2

            intensityFactor = lightFactorWallToWall(point1, point2, wall2) * reflectanceFactor

            for tlaIndex, lightAmount in enumerate(point1.personalTLA):
#                if tlaIndex == 0 and len(list1) == 1:
#                    print tlaIndex, lightAmount, t, intensityFactor, point1, point2, wall2

                addToTimeLightArrayObject(convertTLAIndexToTime(tlaIndex)+t, lightAmount*intensityFactor, \
                    point2.personalTLA)

#    print list2[0].personalTLA
#    print wall2

#    print list2[0].personalTLA

sourcePoint = Point(0,0,0)

# z = 0
reflectorWall = Wall(0,0,1,0, np.array([1,0,0]), np.array([0,1,0]), lambertianReflectance=0.5, \
    specularReflectance=0.01, name="Reflector")

# x + y + z = 1
targetWall = Wall(1,1,1,1, np.array([-2,1,1]), np.array([-2,1,1]), lambertianReflectance=0.5, \
    specularReflectance=0.01, name="Target")

# z = 1
detectorWall = Wall(0,0,1,1, np.array([1,0,0]), np.array([0,1,0]), lambertianReflectance=0.5, \
    specularReflectance=0.01, name="Detector")

detectorPoint = Point(-1,-1,1)

multipleBounceProblemWallPointDetector([targetWall, reflectorWall, detectorWall],
                                        [np.array([0.33, 0.33, 0.34]), \
                                         np.array([0,0,0]), \
                                         np.array([-1,-1,1])], \
                                        sourcePoint, detectorPoint, detectorWall)

print detectorPoint.personalTLA

axes = p.gca()
axes.set_ylim([0, 1500])

plotArray(detectorPoint.personalTLA)
p.savefig("part_specular.png")
p.show()
