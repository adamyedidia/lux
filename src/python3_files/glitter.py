from scenegenerator import Circle, Patch, Room, getSphereAnglesFromXUnitVector
from math import acos, sqrt, cos, sin, asin, tan, atan, pi
import numpy as np
import random
import pickle
import time

# X is the camera-to-cloud dimension
# Z is the dimension of gravity

# All areas measured in ps2 ~= mm2/10
# All times/distances in ps
# Recall that 1ps = 0.3mm

# Measured in ps2
GLITTER_AREA = 10

# measured in ps
TIME_RESOLUTION = 400

# Measured in ps2
PIXEL_AREA = 10

PIXELS_TO_A_SIDE = 32

# ~1m = 3ns
DISTANCE_TO_CLOUD = 3e3

# 100 ns
DETECTOR_COOLDOWN = 1e5

# 100 us
ROTATION_PERIOD = 1e8

# in radians
THETA_ROTATION_AMT_PER_PERIOD = 1e-3
PHI_ROTATION_AMT_PER_PERIOD = 0.01

# 1 m/s
DIST_PER_PERIOD_AVG = -0.3
DIST_PER_PERIOD_STDEV = 0.03

PHOTONS_PER_PULSE = 1e10

# 2 seconds
EXPERIMENT_DURATION = 1e8

MAX_WALL_BRIGHTNESS = 0.1

GLITTER_SPECULARITY = 0.9
GLITTER_LAMBERTIANITY = 0.01

NUM_GLITTER_FLAKES = 1e3

PULSE_AREA = 1e6

# ~1/3 of a meter
PULSE_HEIGHT = 1e3
PULSE_WIDTH = 1e3

PIXEL_REGION_HEIGHT = float(PULSE_WIDTH) / PIXELS_TO_A_SIDE
PIXEL_REGION_WIDTH = float(PULSE_WIDTH) / PIXELS_TO_A_SIDE

GLITTER_CLOUD_STDEV = 250
#
OBJECT_RADIUS_AVG = 1
OBJECT_RADIUS_STDEV = 0.2

# Absolutely NO idea
NOISE_PIXELS_PER_PICOSECOND = 1e-10

CAMERA_LOCATION = np.array([-DISTANCE_TO_CLOUD, 0, 0])

#INITIAL_CLOUD_CENTER = np.array([0, 0, PULSE_HEIGHT/2.])
INITIAL_CLOUD_CENTER = np.array([0,0,0])

PIXEL_SIDE_LENGTH = sqrt(PIXEL_AREA)

CAMERA_SIDE_LENGTH = PIXELS_TO_A_SIDE*PIXEL_SIDE_LENGTH

# Got rid of a factor of 2 there just to be more conservative; plus there  could
# weird distance effects or hypotenuse effects that nudge it above 1/2
ANGLE_OF_GLARE = PIXEL_SIDE_LENGTH / DISTANCE_TO_CLOUD

experimentParams = {
    "GLITTER_AREA": GLITTER_AREA,
    "TIME_RESOLUTION": TIME_RESOLUTION,
    "PIXEL_AREA": PIXEL_AREA,
    "PIXELS_TO_A_SIDE": PIXELS_TO_A_SIDE,
    "DISTANCE_TO_CLOUD": DISTANCE_TO_CLOUD,
    "DETECTOR_COOLDOWN": DETECTOR_COOLDOWN,
    "ROTATION_PERIOD": ROTATION_PERIOD,
    "THETA_ROTATION_AMT_PER_PERIOD": THETA_ROTATION_AMT_PER_PERIOD,
    "PHI_ROTATION_AMT_PER_PERIOD": PHI_ROTATION_AMT_PER_PERIOD,
    "DIST_PER_PERIOD_AVG": DIST_PER_PERIOD_AVG,
    "DIST_PER_PERIOD_STDEV": DIST_PER_PERIOD_STDEV,
    "PHOTONS_PER_PULSE": PHOTONS_PER_PULSE,
    "EXPERIMENT_DURATION": EXPERIMENT_DURATION,
    "MAX_WALL_BRIGHTNESS": MAX_WALL_BRIGHTNESS,
    "GLITTER_SPECULARITY": GLITTER_SPECULARITY,
    "GLITTER_LAMBERTIANITY": GLITTER_LAMBERTIANITY,
    "NUM_GLITTER_FLAKES": NUM_GLITTER_FLAKES,
    "PULSE_AREA": PULSE_AREA,
    "PULSE_HEIGHT": PULSE_HEIGHT,
    "PULSE_WIDTH": PULSE_WIDTH,
    "PIXEL_REGION_HEIGHT": PIXEL_REGION_HEIGHT,
    "PIXEL_REGION_WIDTH": PIXEL_REGION_WIDTH,
    "GLITTER_CLOUD_STDEV": GLITTER_CLOUD_STDEV,
    "OBJECT_RADIUS_AVG": OBJECT_RADIUS_AVG,
    "OBJECT_RADIUS_STDEV": OBJECT_RADIUS_STDEV,
    "NOISE_PIXELS_PER_PICOSECOND": NOISE_PIXELS_PER_PICOSECOND,
    "CAMERA_LOCATION": CAMERA_LOCATION,
    "INITIAL_CLOUD_CENTER": INITIAL_CLOUD_CENTER,
    "PIXEL_SIDE_LENGTH": PIXEL_SIDE_LENGTH,
    "CAMERA_SIDE_LENGTH": CAMERA_SIDE_LENGTH,
    "ANGLE_OF_GLARE": ANGLE_OF_GLARE
}

def pront(x):
    print(x)

# The z-dimension is the dimension of azimuthal rotation here.
def getZUnitVectorFromSphereAngles(phi, theta):
    x = sin(phi)*cos(theta)
    y = sin(phi)*sin(theta)
    z = cos(phi)

    return np.array([x,y,z])

def getSphereAnglesFromZUnitVector(unitVector):
    phi = acos(unitVector[2]) # arccos(z)

    theta = acos(unitVector[0]/sin(phi))

    return phi, theta

def createCloudOfGlitter(numFlakes):
    return [Flake() for _ in range(int(numFlakes))]

def normalize(vec):
    return vec/np.linalg.norm(vec)

class Flake:
    def __init__(self):
        self.orientationVec = np.random.random() - np.array([0.5]*3)
        # normalize it
        self.orientationVec = self.orientationVec / np.linalg.norm(self.orientationVec)

        self.phi, self.theta = getSphereAnglesFromZUnitVector(self.orientationVec)

        self.thetaPerPeriod = np.random.normal(0., THETA_ROTATION_AMT_PER_PERIOD)
        self.phiPerPeriod = np.random.normal(0., PHI_ROTATION_AMT_PER_PERIOD)

        self.zPerPeriod = np.random.normal(DIST_PER_PERIOD_AVG, DIST_PER_PERIOD_STDEV)

        self.position = np.array([np.random.normal(i, GLITTER_CLOUD_STDEV) for \
            i in INITIAL_CLOUD_CENTER])

    def rotateAndFall(self):
        self.theta += self.thetaPerPeriod
        self.phi += self.phiPerPeriod

        if self.phi > pi:
            self.phi += 2*(pi-self.phi)
            self.theta += pi

        if self.phi < 0:
            self.phi *= -1
            self.theta += pi

        if self.theta > 2*pi:
            self.theta -= 2*pi
        if self.theta < 0:
            self.theta += 2*pi

        self.orientationVec = getZUnitVectorFromSphereAngles(self.phi, self.theta)

        self.position[2] += self.zPerPeriod

    def __str__(self):
        return "Flake at " + str(self.position) + " with orientation phi = " + \
            str(self.phi) + " and theta = " + str(self.theta)

class CameraPixel:
    def __init__(self, position, fakePhotonsPerPicosecond):
        self.position = position
        self.fakePhotonsPerPicosecond = fakePhotonsPerPicosecond
        self.returns = []

    def addReturn(self, ret):
        self.returns.append(ret)

    def clearReturns(self):
        self.returns = []

    def addRandomReturn(self, maxTime):
        self.addReturn((1, random.random()*maxTime), 2*pi*random.random())

    def addReturnsFromNoise(self):
        lastestReturnTime = -1

        for ret in self.returns:
            # ret[1][1] is second return time
            if ret[1][1] > lastestReturnTime:
                lastestReturnTime = ret[1][1]

        numFakePhotons = np.random.poisson(latestReturnTime*fakePhotonsPerPicosecond)

        for _ in range(numFakePhotons):
            self.addRandomReturn(lastestReturnTime)

    def collapseReturns(self):
        self.returns.sort(key=lambda x: x[1]) # sort by return time

        for ret in self.returns:
            if random.random() < ret[0]:
                return (ret[1], ret[2])

        return (float("Inf"), 0.)


    def getReading(self):
        self.addReturnsFromNoise()
        reading = self.collapseReturns()
        self.clearReturns()

        return reading

class Camera:
    def __init__(self):
        cameraNegativeCornerLocation = CAMERA_LOCATION + \
            np.array([0, -CAMERA_SIDE_LENGTH/2., -CAMERA_SIDE_LENGTH/2.])

        self.pixelArray = []

        for i in range(PIXELS_TO_A_SIDE):
            self.pixelArray.append([])
            for j in range(PIXELS_TO_A_SIDE):
                pixelLocation = cameraNegativeCornerLocation + \
                    np.array([0, (i+0.5)*PIXEL_SIDE_LENGTH, \
                    (j+0.5)*PIXEL_SIDE_LENGTH])

                self.pixelArray[-1].append(CameraPixel(pixelLocation, \
                    NOISE_PIXELS_PER_PICOSECOND))

    def getPixelAssociatedWithLocation(self, location):
        distanceFactor = (location[0]+DISTANCE_TO_CLOUD) / float(DISTANCE_TO_CLOUD)

        pixelY = location[1]/distanceFactor
        pixelZ = location[2]/distanceFactor

        if (pixelY > PULSE_WIDTH / 2.) or \
            (pixelY < -PULSE_WIDTH / 2.) or \
            (pixelZ > PULSE_HEIGHT / 2.) or \
            (pixelZ < -PULSE_HEIGHT / 2.):

            return None

        effectiveY = pixelY + CAMERA_SIDE_LENGTH / 2.
        effectiveZ = pixelZ + CAMERA_SIDE_LENGTH / 2.

        yByPixelIndex = int(effectiveY * PIXELS_TO_A_SIDE / float(PULSE_WIDTH))
        zByPixelIndex = int(effectiveZ * PIXELS_TO_A_SIDE / float(PULSE_HEIGHT))

        return self.pixelArray[yByPixelIndex][zByPixelIndex]

def reflectVectorThroughSurface(vector, surfaceNormal):
    cosTheta = np.linalg.norm(np.dot(vector, surfaceNormal))

    return -2*cosTheta*surfaceNormal + vector

def emitPulse(cloudOfGlitter, camera, room):
    flakeResponseCount = 0

    for flake in cloudOfGlitter:
        associatedPixel = camera.getPixelAssociatedWithLocation(flake.position)

        if associatedPixel != None:
            flakeResponseCount += 1

            beamVector = flake.position - associatedPixel.position
            cameraToFlakeDistance = np.linalg.norm(beamVector)
            incomingAngle = acos(abs(np.dot(beamVector/cameraToFlakeDistance, \
                flake.orientationVec)))
            distanceFactor = cameraToFlakeDistance / float(DISTANCE_TO_CLOUD)
            visibleGlitterArea = cos(incomingAngle)*GLITTER_AREA
            areaFraction = visibleGlitterArea / \
                (distanceFactor*PULSE_WIDTH*PULSE_HEIGHT)

            photonsHittingFlakeFirstBounce = areaFraction*PHOTONS_PER_PULSE

            if incomingAngle < ANGLE_OF_GLARE:
                # Aaaah, the light! It blinds me!!
                photonsFirstReturn = photonsHittingFlakeFirstBounce*GLITTER_SPECULARITY
                timeOfFirstReturn = 2*cameraToFlakeDistance
                photonsSecondReturn = 0.
                timeOfSecondReturn = 4*cameraToFlakeDistance # irrelevant
                flakeTheta = 2*pi*random.random()

            else:
                photonsFirstReturn = photonsHittingFlakeFirstBounce * GLITTER_LAMBERTIANITY * \
                    PIXEL_AREA/(2*pi*cameraToFlakeDistance*cameraToFlakeDistance)

                vectorOutward = reflectVectorThroughSurface(beamVector, flake.orientationVec)
                flakePhi, flakeTheta = getSphereAnglesFromXUnitVector(normalize(vectorOutward))
                scenePatch = room.getClosestOccluderInDirection(flake.position,
                    flakePhi, flakeTheta)

                photonsHittingScenePatch = photonsHittingFlakeFirstBounce * \
                    GLITTER_SPECULARITY

                flakeToSceneVector = scenePatch.position - flake.position
                flakeToSceneDistance = np.linalg.norm(flakeToSceneVector)

                photonsComingBackToGlitterFlake = scenePatch.albedo * \
                    visibleGlitterArea / (2*pi*flakeToSceneDistance*flakeToSceneDistance)

                photonsSecondReturn = photonsComingBackToGlitterFlake * GLITTER_SPECULARITY

                timeOfFirstReturn = 2*cameraToFlakeDistance
                timeOfSecondReturn = 2*cameraToFlakeDistance + 2*flakeToSceneDistance

            associatedPixel.addReturn((photonsFirstReturn, timeOfFirstReturn, \
                2*pi*random.random()))
            associatedPixel.addReturn((photonsSecondReturn, timeOfSecondReturn, flakeTheta))

    pront("Flakes responded: " + str(flakeResponseCount) + "/" + str(int(NUM_GLITTER_FLAKES)))

    return [[pixel.collapseReturns() for pixel in pixelRow] for pixelRow in \
        camera.pixelArray]




#            outgoingLightVector = flake.
#            flakePhi, flakeTheta = getSphereAnglesFromXUnitVector()
#            scenePoint = room.getClosestOccluderInDirection(flake.position, \
#                )


#if False:
if __name__ == "__main__":
    f = Flake()

    # Set up the experimental apparatus
    pront("Setting up the experimental appartus...")

    cloudOfGlitter = createCloudOfGlitter(NUM_GLITTER_FLAKES)
    camera = Camera()

    # Generate the scene
    pront("Generating the scene...")

    room = Room(2*DISTANCE_TO_CLOUD)
    room.addManyRandomCircles(5, 0.4*DISTANCE_TO_CLOUD)

    # Begin experiment
    pront("Beginning the experiment!")

    listOfData = []
    currentMinilist = []

    currentTime = 0.

    nextRotationTime = ROTATION_PERIOD

    t = time.time()

    while currentTime < EXPERIMENT_DURATION:
        currentMinilist.append(emitPulse(cloudOfGlitter, camera, room))



        currentTime += DETECTOR_COOLDOWN

        pront("Progress towards finish: " + str(100*currentTime/EXPERIMENT_DURATION) + "%")
        pront("Last step took " + str(time.time() - t) + " seconds.")
        t = time.time()

        if currentTime > nextRotationTime:
            [flake.rotateAndFall() for flake in cloudOfGlitter]
            nextRotationTime += ROTATION_PERIOD

            listOfData.append(currentMinilist)
            currentMinilist = []

            pront("Done with batch!")

    pickle.dump((listOfData, experimentParams), open("glitter_data.p", "w"))
