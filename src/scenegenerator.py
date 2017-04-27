from math import cos, sin, ceil, floor, atan2, sqrt, pi, acos
import random
import numpy as np
from PIL import Image

def lambertian_brdf(incidentPhi, normalVector, outgoingPhi, incidentOutgoingTheta):
    if outgoingPhi < pi/2:
        return cos(incidentPhi)/(2*pi)
    else:
        return 0

class Patch:
    def __init__(self, position=np.array([0,0,0]), normal=np.array([1,0,0]), area=0, albedo=0.1, brdf=lambertian_brdf):
        self.position = position # x y z
        self.normal = normal # unit vector, surface normal, orientation
        self.area = area # surface area
        self.albedo = albedo # reflectivity normalization
        self.brdf = brdf # normalized brdf


MAX_ALBEDO = 0.1

CARDINAL_DIRECTIONS = [np.array([0,0,1]),
                        np.array([0,0,-1]),
                        np.array([0,1,0]),
                        np.array([0,-1,0]),
                        np.array([1,0,0]),
                        np.array([-1,0,0])]

# Generates spherical scenes
# A scene consists of a bunch of circles and squares of varying reflectances, distances,
# and orientations

# This is relative to looking straight ahead in the (1,0,0) direction
# phi makes you look away from the center (0 <= phi <= pi)
# theta makes you rotate your field of view around the center (0 <= theta <= 2pi)
def getXUnitVectorFromSphereAngles(phi, theta):
    x = cos(phi)
    y = sin(phi)*cos(theta)
    z = sin(phi)*sin(theta)

    return np.array([x,y,z])

def getSphereAnglesFromXUnitVector(unitVector):
    phi = acos(unitVector[0]) # arccos(x)
    theta = acos(unitVector[1]/sin(phi))

    return phi, theta

def getPlanesDescribingRayVector(pointOfOrigin, rayVector):
    randomVector = np.random.random(3)

    plane1Coeffs = np.cross(rayVector, randomVector)
    plane1Intercept = np.dot(plane1Coeffs, pointOfOrigin)

    plane2Coeffs = np.cross(rayVector, plane1Coeffs)
    plane2Intercept = np.dot(plane2Coeffs, pointOfOrigin)

    return plane1Coeffs, plane1Intercept, plane2Coeffs, plane2Intercept

def extractListFromNPArray(nparray):
    return [i for i in nparray]

def matrixify(listOfNpArrays):
    return np.array([extractListFromNPArray(i) for i in listOfNpArrays])

class Circle:
    def __init__(self, center, radius, normal, albedo):
        self.radius = radius
        self.normal = normal
        self.center = center

        self.albedo = albedo

        self.intercept = np.dot(self.normal, self.center)

    def doesItOcclude(self, pointOfOrigin, rayVector):
        plane1Coeffs, plane1Intercept, plane2Coeffs, plane2Intercept = \
            getPlanesDescribingRayVector(pointOfOrigin, rayVector)

        coeffMatrix = matrixify([plane1Coeffs, plane2Coeffs, self.normal])
        intercepts = np.array([plane1Intercept, plane2Intercept, self.intercept])

        try:
            solutionPoint = np.linalg.solve(coeffMatrix, intercepts)
        except np.linalg.linalg.LinAlgError:
            # Singular matrix error; implies that rayVector is parallel to the circle
            return False

        # If it's within the circle, and in the right direction,
        # return the solutionPoint as a patch
        if np.linalg.norm(solutionPoint - self.center) < self.radius and \
            (solutionPoint - pointOfOrigin)[0]/float(rayVector[0]) > 0:

            return Patch(solutionPoint, self.normal, 0, self.albedo)

        # if it's not within the circle, return False
        return False


class Room:
    def addCircle(self, center, radius, normal, albedo):
        self.listOfObjects.append(Circle(center, radius, normal, albedo))

    # A cubical room with walls of random albedos
    def __init__(self, roomSideLength):
        self.listOfObjects = []
        self.roomSideLength = roomSideLength

        # Make the room walls
        for direction in CARDINAL_DIRECTIONS:
            center = roomSideLength/2*direction
            self.addCircle(center, self.roomSideLength, direction,
                MAX_ALBEDO*random.random())

    def getRandomPointInRoom(self):
        return (np.random.random()-np.array([0.5, 0.5, 0.5]))*self.roomSideLength

    # X is the special direction here
    def getClosestOccluderInDirection(self, pointOfOrigin, phi, theta):
        smallestDistance = float("Inf")
        nearestPatch = None

        rayVector = getXUnitVectorFromSphereAngles(phi, theta)
        for obj in self.listOfObjects:
            occlusionResult = obj.doesItOcclude(pointOfOrigin, rayVector)
            if not (occlusionResult == False):
                distance = np.linalg.norm(occlusionResult.position - pointOfOrigin)

                if distance < smallestDistance:
                    smallestDistance = distance
                    nearestPatch = occlusionResult

        return nearestPatch

    def takePictureOfRoom(self, cameraLocation, cameraVector, cameraArc, pixelsToASide,
        filename):
        photo = Image.new("L", (pixelsToASide, pixelsToASide))

        listOfPixels = []

        for x in range(-int(floor(pixelsToASide/2)),int(ceil(pixelsToASide/2))):
            print x
            for y in range(-int(floor(pixelsToASide/2)),int(ceil(pixelsToASide/2))):
                theta = atan2(y,x)
                phi = 2.*sqrt(x*x + y*y)*cameraArc/pixelsToASide

                closestOccluder = self.getClosestOccluderInDirection(cameraLocation, \
                    theta, phi)

                listOfPixels.append(ceil(closestOccluder.albedo/MAX_ALBEDO*256-1))

        photo.putdata(listOfPixels)

        photo.save(filename)

    def addRandomCircle(self, radius):
        centerLocation = self.getRandomPointInRoom()
        theta = random.random()*2*pi
        phi = random.random()*pi
        normalVec = getXUnitVectorFromSphereAngles(theta, phi)
        albedo = random.random()*MAX_ALBEDO

        self.addCircle(centerLocation, radius, normalVec, albedo)

    def addManyRandomCircles(self, numCircles, radius):
        for _ in range(numCircles):
            self.addRandomCircle(radius)

#if False:
if __name__ == "__main__":

    room = Room(10)
    room.addManyRandomCircles(5, 2)

    room.takePictureOfRoom(np.array([0,0,0]), np.array([1,0,0]), pi/3, 100,
        "empty_room_picture.png")


#def generateScene(objectReflectanceAvg, objectSizeStdev, \
#                    objectDistanceAvg, objectDistanceStdev, \
#                    )
