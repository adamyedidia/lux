from __future__ import division
import pylab
import imageio
import numpy as np
from math import acos, pi, cos, sin
import matplotlib.pyplot as p
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def normalize(v):
    return v / np.linalg.norm(v)

filename = 'ceiling_1_low_res.m4v'
vid = imageio.get_reader(filename,  'ffmpeg')

#nums = [10, 43, 90, 130, 170, 210, 287]

print np.array(vid).shape

numFrames = len(vid)

averageFrame = None

for i in range(int(numFrames)):
    im = vid.get_data(i)
    if averageFrame == None:
        print "hi"
        averageFrame = np.array(im).astype(float)
    else:
        averageFrame = averageFrame + np.array(im).astype(float)


#print averageFrame
averageFrame = averageFrame / numFrames

#print averageFrame[350][600]
#print averageFrame.astype(int)[350][600]

#firstFrame = vid.get_data(80)
#print firstFrame[350][600]
#print type(firstFrame[350][600][0])

#pylab.imshow(firstFrame)
#p.show()

#p.clf()
#pylab.imshow(averageFrame.astype(np.uint8))
#pylab.show()
#p.clf()

#(305, 360)
#(548, 278)
#(440, 0)

# Magic numbers
doorToCeilCorner = np.array([548, 278])
alongCeilingVector = normalize(np.array([440, 0]) - doorToCeilCorner)
alongDoorframeVector = normalize(np.array([305, 360]) - doorToCeilCorner)

longestVectorLength = np.dot(np.array([0,0]) - doorToCeilCorner, alongCeilingVector) + \
    0.001

def getAngleBetweenUnitVectors(v1, v2):
    return acos(np.dot(v1, v2))

def getSmoothedDerivative(l, derivWidth):
    return [sum(l[i+derivWidth:i+2*derivWidth]) - sum(l[i:i+derivWidth]) \
        for i in range(len(l) - 2*derivWidth)]

#def extractDistFrom

#frame = np.array(vid.get_data(168)).astype(float)
#frame = np.array(vid.get_data(80)).astype(float)

#print frame.shape

#frame -= averageFrame # subract off average frame


MAKE_PLOTS = True

#print numFrames
#pylab.imshow(frame)
#pylab.show()

def extractYFromDBlueDDist(dBlueDDist):
    # Magic number
    # I don't think any peaks beyond 60 are plausible
    bestY = -1
    bestDistDeriv = 0
    for y, distDeriv in enumerate(dBlueDDist[:60]):
        if distDeriv > bestDistDeriv:
            bestDistDeriv = distDeriv
            bestY = y

    return bestY

def extractThetaFromDBlueDTheta(dBlueDTheta):
    bestTheta = -1
    # magic number
    bestThetaDeriv = -15

    # another magic number :(
    thetaPush = 25
    for thetaIndex, thetaDeriv in enumerate(dBlueDTheta[thetaPush:]):
        if thetaDeriv < bestThetaDeriv:
            # less than is intentional; we want the angle where the blue stuff stops
            bestThetaDeriv = thetaDeriv
            bestTheta = (thetaIndex+thetaPush)/NUM_THETA_BINS * pi/2

#            print thetaIndex, bestTheta


    return bestTheta

NUM_THETA_BINS = 200
NUM_DIST_BINS = 200

CLUMP_SIZE = 10
DERIVATIVE_SMOOTHING = 10

frameClumpSum = np.array(vid.get_data(1)).astype(float)*0

prism = cm.get_cmap(name="gist_rainbow")
norm = Normalize(0, numFrames)
scalarMappable = cm.ScalarMappable(norm=norm, cmap=prism)

p.clf()

# now we see the bluishness of the frame
for frameNum in range(int(numFrames)):
    if frameNum % CLUMP_SIZE == CLUMP_SIZE-1:
        frameClumpSum += np.array(vid.get_data(frameNum)).astype(float)
        frame = frameClumpSum / CLUMP_SIZE
        frameClumpSum *= 0

        print frameNum
#        rawFrame = np.array(vid.get_data(frameNum))
#        frame = rawFrame.astype(float)

        frameForViewing = np.copy(frame)


        frame -= averageFrame


        thetaBins = [0]*NUM_THETA_BINS
        thetaCounts = [0]*NUM_THETA_BINS

        distBins = [0]*NUM_DIST_BINS
        distCounts = [0]*NUM_DIST_BINS

        for y, row in enumerate(frame):
    #        print y
            for x, pixel in enumerate(row):

                vecFromCornerToPoint = np.array([x, y]) - doorToCeilCorner
                lengthOfVecFromCornerToPoint = np.linalg.norm(vecFromCornerToPoint)
                normalizedVecFromCornerToPoint = vecFromCornerToPoint / \
                    lengthOfVecFromCornerToPoint

                theta = getAngleBetweenUnitVectors(normalizedVecFromCornerToPoint, \
                    alongDoorframeVector)

                cosThetaDist = np.dot(normalizedVecFromCornerToPoint, alongCeilingVector)
                dist = cosThetaDist * lengthOfVecFromCornerToPoint


                # first clause makes sure you don't have anything beyond the line
                # second clause makes sure you don't have anything not on the ceiling
                if (theta < pi/2 and theta >= 0) and dist > 0:

                    # r and g are irrelevant

                    if pixel[2] < 0:
                        frameForViewing[y][x][0] = min(-pixel[2]*10, 255)
                        frameForViewing[y][x][1] = min(-pixel[2]*10, 255)
                        frameForViewing[y][x][2] = 0

                    else:
                        frameForViewing[y][x][0] = 0
                        frameForViewing[y][x][1] = 0
                        frameForViewing[y][x][2] = min(pixel[2]*10, 255)

            #        print pixel[2]

                    thetaIndex = int(theta/(pi/2)*NUM_THETA_BINS)
                    thetaBins[thetaIndex] += pixel[2]
                    thetaCounts[thetaIndex] += 1


                    distIndex = int(dist/longestVectorLength*NUM_DIST_BINS)
                    distBins[distIndex] += pixel[2]
                    distCounts[distIndex] += 1

    #    frameForViewingUInt8 = frameForViewing.astype(np.uint8)

        thetaBins = [i/j for i,j in zip(thetaBins, thetaCounts)]
        distBins = [i/j for i,j in zip(distBins, distCounts)]

        dBlueDTheta = getSmoothedDerivative(thetaBins, DERIVATIVE_SMOOTHING)
        dBlueDDist = getSmoothedDerivative(distBins, DERIVATIVE_SMOOTHING)

#        r = np.array([i[:2] for i in frameForViewing[:2]]).astype(np.uint8)

#        pylab.imshow(np.array([i[:2] for i in frameForViewing[:2]]).astype(np.uint8).copy())
#        pylab.show()
#        pylab.imshow(np.array([i[:2] for i in frameForViewing[:2]]).astype(np.uint8))

        if MAKE_PLOTS:

            p.clf()
            pylab.imshow(frameForViewing.astype(np.uint8))
            pylab.show()
            pylab.imshow(frameForViewing.astype(np.uint8))
            p.savefig("color_adjusted_" + str(frameNum) + ".png")

            p.clf()
            p.plot(thetaBins)
            p.savefig("theta_bins_" + str(frameNum) + ".png")
    #        p.show()

            p.clf()
            p.plot(dBlueDTheta)
            p.savefig("d_blue_d_theta_" + str(frameNum) + ".png")
    #        p.show()

            p.clf()
            p.plot(distBins)
            p.savefig("dist_bins_" + str(frameNum) + ".png")
    #        p.show()

            p.clf()
            p.plot(dBlueDDist)
            p.savefig("d_blue_d_dist_" + str(frameNum) + ".png")
    #        p.show()

        else:
            y = extractYFromDBlueDDist(dBlueDDist)
            theta = extractThetaFromDBlueDTheta(dBlueDTheta)

            if y != -1 and theta != -1:
                print theta, cos(theta), sin(theta)
                x = y*cos(theta)/sin(theta)
                p.plot(x, y, marker="o", color=scalarMappable.to_rgba(frameNum))
                print (x, y)

    else:
        frameClumpSum += np.array(vid.get_data(frameNum)).astype(float)

p.savefig("movement_reconstruction.png")
p.show()


#    image = vid.get_data(num)
#    print im
#    fig = pylab.figure()
#    pylab.imshow(im)
#    pylab.show()
