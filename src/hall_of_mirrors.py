from __future__ import division
import numpy as np
from math import sqrt, tan, pi, cos, sin
import matplotlib.pyplot as pl

def getABFromP1P2(p1, p2):
    x1 = p1[0][0]
    y1 = p1[1][0]
    x2 = p2[0][0]
    y2 = p2[1][0]

    a = (y2 - y1)/(x1*y2 - x2*y1)
    b = (x2 - x1)/(y1*x2 - y2*x1)

    return a, b

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

class Room:
    def __init__(self, listOfMirrors):
        self.listOfMirrors = listOfMirrors

    def isInRoom(self, p):
        for mirror in self.listOfMirrors:
            if not mirror.isFacing(p):
                return False
        return True

    def countReflections(self, p):
        numReflections = 0

        for mirror in self.listOfMirrors:
            if mirror.pathIntersects(p, mirror.reflectPoint(p)):
                numReflections += 1

        return numReflections

    def countReflectionsBetween(self, p1, p2):
        numReflections = 0

        for mirror in self.listOfMirrors:
            if mirror.pathIntersects(p1, mirror.reflectPoint(p2)):
                numReflections += 1

        return numReflections

class Mirror:
    def __init__(self, start, end, facing):
        self.start = start
        self.end = end
        self.startToEnd = end - start
        self.length = np.linalg.norm(self.startToEnd)
        self.facing = facing

        self.a, self.b = getABFromP1P2(start, end)

    def __str__(self):
        return "Mirror(" + str(self.start) + "," + str(self.end) + ") " + \
            "facing " + str(self.facing)

    def reflectPoint(self, point):
        startToPoint = point - self.start
        startToPointProj = self.start + self.startToEnd * \
            np.dot(np.transpose(startToPoint), self.startToEnd) \
            / np.linalg.norm(self.startToEnd)**2

        pointToPointProj = startToPointProj - point
        return point + 2*pointToPointProj

    def pathIntersects(self, p1, p2):
        a, b = getABFromP1P2(p1, p2)

        x, y = getABFromP1P2(np.array([[a], [b]]), np.array([[self.a], [self.b]]))
        intersectionPoint = np.array([[x], [y]])

        if (distance(intersectionPoint, self.start) < self.length) and \
            (distance(intersectionPoint, self.end) < self.length):

            return True
        return False

    def isFacing(self, p):
#        print np.transpose(p - self.start), self.facing
        return (np.dot(np.transpose(p - self.start), self.facing) > 0)

def createRegularNGon(n, A):
    s = sqrt(4*A*tan(pi/n)/n)

    currentLocation = np.array([[1.],[1.]])

    rotationAngle = pi - (n-2)*pi/n
    rotationMatrix = np.array([[cos(rotationAngle), -sin(rotationAngle)],
        [sin(rotationAngle), cos(rotationAngle)]])

    movementVector = np.array([[s], [0.]])
    facingVector = np.array([[0.],[1.]])

    listOfMirrors = []

    for _ in range(n):
        newLocation = currentLocation + movementVector

        listOfMirrors.append(Mirror(currentLocation, \
            newLocation, facingVector))

#        print currentLocation
#        print movementVector
#        print facingVector

        currentLocation = np.add(currentLocation, movementVector)
        movementVector = np.dot(rotationMatrix, movementVector)
        facingVector = np.dot(rotationMatrix, facingVector)

#        print listOfMirrors[-1]

    return Room(listOfMirrors)

def showReflectionsInRoom(room, minX, minY, maxX, maxY, step):
    x = minX
    y = minY

    reflectionArray = []

    while x < maxX:
        reflectionArray.append([])
        while y < maxY:
            p = np.array([[x], [y]])

            if room.isInRoom(p):
#                print p
                reflectionArray[-1].append(room.countReflections(p))

            else:
                reflectionArray[-1].append(0)

            y += step
        y = minX
        x += step

#    print reflectionArray

    pl.matshow(np.array(reflectionArray))
#    pl.matshow(np.array([[0,1],[2,3]]))
    pl.show()

def showReflectionsInRoomFromPoint(fixedPoint, room, minX, minY, maxX, maxY, step):
    x = minX
    y = minY

    reflectionArray = []

    while x < maxX:
        reflectionArray.append([])
        while y < maxY:
            p = np.array([[x], [y]])

            if room.isInRoom(p):
#                print p
                reflectionArray[-1].append(room.countReflectionsBetween(p, fixedPoint))

            else:
                reflectionArray[-1].append(0)

            y += step
        y = minX
        x += step

#    print reflectionArray

#    pl.plot(fixedPoint[0], fixedPoint[1], "ko")
    pl.matshow(np.array(reflectionArray))
#    pl.matshow(np.array([[0,1],[2,3]]))
    pl.show()


#m = Mirror(np.array([1,2]), np.array([2.5,3.5]))
#p = np.array([5, 2])

room = createRegularNGon(31, 1)
showReflectionsInRoom(room, 0.555, 0.555, \
    2.445, 2.445, 0.01)
