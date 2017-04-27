from __future__ import division
from PIL import Image
import numpy as np
import matplotlib.pyplot as p
from math import pi, cos, sin, sqrt
from scipy.optimize import fmin_bfgs
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import sys

EPS = 1e-10

learnRate = 1e-5

def isWhite(rgb):
    if rgb[0] == 255:
        if rgb[1] == 255:
            if rgb[2] == 255:
                return True

    return False

def findGreatestContiguousNonWhite(row):
    widthOfContig = 0
    widestContig = 0
    startIndex = None
    widestContigStartIndex = None

    for i, color in enumerate(row):
        if isWhite(color):
            widthOfContig = 0
            startIndex = None
        else:
            if startIndex == None:
                startIndex = i

            widthOfContig += 1
            if widthOfContig > widestContig:
                widestContig = widthOfContig
                widestContigStartIndex = startIndex

    return widestContigStartIndex, widestContig

def extractWidthsAndCenters(imgArray):

    centers = []
    widths = []

    for row in arr:
        startIndex, width = findGreatestContiguousNonWhite(row)

    # hack for crappily-filled-in images
        if startIndex == None:
            if len(centers) == 0:
                centers.append(-1)
                widths.append(-1)
            else:
                centers.append(centers[-1])
                widths.append(widths[-1])

        else:
            centers.append(startIndex + width/2)
            widths.append(width)

    # hack for crappily-filled-in images
    for i in range(len(centers)-1, -1, -1):
        if centers[i] == -1:
            centers[i] = centers[i+1]
            widths[i] = width[i+1]

        if widths[i] < 10:
            widths[i] = 10

    return centers, widths
#    p.plot(centers)
#    p.show()

#    p.clf()
#    p.plot(widths)
#    p.show()

def widthToR(width):
    return 1./width

def centerToTheta(center):
    return center/imageWidth*pi/2

def average(l):
    return sum(l)/len(l)

def computeAverageSpeed(x):
    global averageSpeed

    thetas = x[:imageHeight]
    rs = x[imageHeight:]

    rTheta = zip(rs, thetas)
    xs = [r*cos(theta) for r, theta in rTheta]
    ys = [r*sin(theta) for r, theta in rTheta]

    dxs = [xs[i+1] - xs[i] for i in range(imageHeight-1)]
    dys = [ys[i+1] - ys[i] for i in range(imageHeight-1)]

    accels = [(dxs[i+1]-dxs[i])**2 + (dys[i+1]-dys[i])**2 for i in \
        range(imageHeight-2)]

    speeds = [sqrt(dx*dx + dy*dy) for dx, dy in zip(dxs, dys)]
    averageSpeed = average(speeds)

    print "avg", averageSpeed

    return averageSpeed

IMAGE_NAME = sys.argv[1]

img = Image.open(IMAGE_NAME)
arr = np.array(img)

imageHeight = arr.shape[0]
imageWidth = arr.shape[1]

centers, widths = extractWidthsAndCenters(IMAGE_NAME)

#x0 = np.array([centerToTheta(i) for i in centers] + \
#    [widthToR(i) for i in widths])

x0 = np.array([pi/4]*imageHeight + [0.1]*imageHeight)

averageSpeed = computeAverageSpeed(x0)

R_WIDTH_PENALTY = 0.#0.01
CENTER_THETA_PENALTY = 1.
ACCEL_PENALTY = 1e3#50.
SPEED_PENALTY = 1000.#0.1

def getSpeedPenalty(theta1, theta2, r1, r2):
    global averageSpeed

    x1 = r1*cos(theta1)
    y1 = r1*sin(theta1)
    x2 = r2*cos(theta2)
    y2 = r2*sin(theta2)

#    print (sqrt((x2-x1)**2 + (y2-y1)**2)-averageSpeed)**2*SPEED_PENALTY

    return (sqrt((x2-x1)**2 + (y2-y1)**2)-averageSpeed)**2*SPEED_PENALTY

def getAccelPenalty(theta1, theta2, theta3, r1, r2, r3):
    x1 = r1*cos(theta1)
    y1 = r1*sin(theta1)
    x2 = r2*cos(theta2)
    y2 = r2*sin(theta2)
    x3 = r3*cos(theta3)
    y3 = r3*sin(theta3)

    vx12 = x2-x1
    vy12 = y2-y1
    vx23 = x3-x2
    vy23 = y3-y2

    return ((vx23-vx12)**2 + (vy23-vy12)**2)*ACCEL_PENALTY

def varySpeedPenaltyWithAndWithoutEps(theta1, theta2, r1, r2, indexVary):
    speedPenaltyWoEps = getSpeedPenalty(theta1, theta2, r1, r2)

    if indexVary == 0:
        speedPenaltyWithEps = getSpeedPenalty(theta1+EPS, theta2, r1, r2)

    if indexVary == 1:
        speedPenaltyWithEps = getSpeedPenalty(theta1, theta2+EPS, r1, r2)

    if indexVary == 2:
        speedPenaltyWithEps = getSpeedPenalty(theta1, theta2, r1+EPS, r2)

    if indexVary == 3:
        speedPenaltyWithEps = getSpeedPenalty(theta1, theta2, r1, r2+EPS)

    return speedPenaltyWithEps - speedPenaltyWoEps

def varyAccelPenaltyWithAndWithoutEps(theta1, theta2, theta3, r1, r2, r3, indexVary):
    accelPenaltyWoEps = getAccelPenalty(theta1, theta2, theta3, \
        r1, r2, r3)

    if indexVary == 0:
        accelPenaltyWithEps = getAccelPenalty(theta1+EPS, theta2, theta3, \
            r1, r2, r3)
    elif indexVary == 1:
        accelPenaltyWithEps = getAccelPenalty(theta1, theta2+EPS, theta3, \
            r1, r2, r3)
    elif indexVary == 2:
        accelPenaltyWithEps = getAccelPenalty(theta1, theta2, theta3+EPS, \
            r1, r2, r3)
    elif indexVary == 3:
        accelPenaltyWithEps = getAccelPenalty(theta1, theta2, theta3, \
            r1+EPS, r2, r3)
    elif indexVary == 4:
        accelPenaltyWithEps = getAccelPenalty(theta1, theta2, theta3, \
            r1, r2+EPS, r3)
    elif indexVary == 5:
        accelPenaltyWithEps = getAccelPenalty(theta1, theta2, theta3, \
            r1, r2, r3+EPS)

    return accelPenaltyWithEps - accelPenaltyWoEps

def penaltyFunc(x):

    thetas = x[:imageHeight]
    rs = x[imageHeight:]

    rTheta = zip(rs, thetas)
    xs = [r*cos(theta) for r, theta in rTheta]
    ys = [r*sin(theta) for r, theta in rTheta]

    dxs = [xs[i+1] - xs[i] for i in range(imageHeight-1)]
    dys = [ys[i+1] - ys[i] for i in range(imageHeight-1)]

    accels = [(dxs[i+1]-dxs[i])**2 + (dys[i+1]-dys[i])**2 for i in \
        range(imageHeight-2)]

    speeds = [sqrt(dx*dx + dy*dy) for dx, dy in zip(dxs, dys)]
    averageSpeed = average(speeds)

    speedsOffAverage = [(speed-averageSpeed)**2 for speed in speeds]

    rOffWidth = [(r-widthToR(width))**2 for r, width in zip(rs, widths)]
    thetaOffCenter = [(theta-centerToTheta(center))**2 for theta, center in \
        zip(thetas, centers)]

    rPenalty = R_WIDTH_PENALTY * sum(rOffWidth)
    thetaPenalty = CENTER_THETA_PENALTY * sum(thetaOffCenter)
    accelPenalty = ACCEL_PENALTY * sum(accels)
    speedPenalty = SPEED_PENALTY * sum(speeds)

    #print rPenalty, thetaPenalty, accelPenalty, speedPenalty

    return rPenalty + thetaPenalty + accelPenalty + speedPenalty

def penaltyFuncPrime(x):
    gradientArray = []

    for i in range(imageHeight):
        # thetas time!
        theta = x[i]
        center = centers[i]
        thetaCenterPenalty = ((theta+EPS-centerToTheta(center))**2 - \
            (theta-centerToTheta(center))**2)*CENTER_THETA_PENALTY

        if i >= 2:
            accelPenaltyBwd = varyAccelPenaltyWithAndWithoutEps(x[i-2], x[i-1], x[i], \
                x[imageHeight+i-2], x[imageHeight+i-1], x[imageHeight+i], 2)

        else:
            accelPenaltyBwd = 0

        if i >= 1 and i < imageHeight-1:
            accelPenaltyHere = varyAccelPenaltyWithAndWithoutEps(x[i-1], x[i], x[i+1], \
                x[imageHeight+i-1], x[imageHeight+i], x[imageHeight+i+1], 1)

        else:
            accelPenaltyHere = 0

        if i < imageHeight-2:
            accelPenaltyFwd = varyAccelPenaltyWithAndWithoutEps(x[i], x[i+1], x[i+2], \
                x[imageHeight+i], x[imageHeight+i+1], x[imageHeight+i+2], 0)

        else:
            accelPenaltyFwd = 0

        if i >= 1:
            speedPenaltyBwd = varySpeedPenaltyWithAndWithoutEps(x[i-1], x[i], \
                x[imageHeight+i-1], x[imageHeight+i], 1)

        else:
            speedPenaltyBwd = 0

        if i < imageHeight-1:
            speedPenaltyFwd = varySpeedPenaltyWithAndWithoutEps(x[i], x[i+1], \
                x[imageHeight+i], x[imageHeight+i+1], 0)

        else:
            speedPenaltyFwd = 0

        gradientArray.append((thetaCenterPenalty + accelPenaltyBwd + accelPenaltyHere + \
            accelPenaltyFwd + speedPenaltyBwd + speedPenaltyFwd)/EPS)

    for i in range(imageHeight):
        # rs time!
        r = x[i+imageHeight]
        width = widths[i]
        rWidthPenalty = ((r+EPS-widthToR(width))**2 - \
            (r-widthToR(width))**2)*R_WIDTH_PENALTY

        if i >= 2:
            accelPenaltyBwd = varyAccelPenaltyWithAndWithoutEps(x[i-2], x[i-1], x[i], \
                x[imageHeight+i-2], x[imageHeight+i-1], x[imageHeight+i], 5)

        else:
            accelPenaltyBwd = 0

        if i >= 1 and i < imageHeight-1:
            accelPenaltyHere = varyAccelPenaltyWithAndWithoutEps(x[i-1], x[i], x[i+1], \
                x[imageHeight+i-1], x[imageHeight+i], x[imageHeight+i+1], 4)

        else:
            accelPenaltyHere = 0

        if i < imageHeight-2:
            accelPenaltyFwd = varyAccelPenaltyWithAndWithoutEps(x[i], x[i+1], x[i+2], \
                x[imageHeight+i], x[imageHeight+i+1], x[imageHeight+i+2], 3)

        else:
            accelPenaltyFwd = 0

        if i >= 1:
            speedPenaltyBwd = varySpeedPenaltyWithAndWithoutEps(x[i-1], x[i], \
                x[imageHeight+i-1], x[imageHeight+i], 3)

        else:
            speedPenaltyBwd = 0

        if i < imageHeight-1:
            speedPenaltyFwd = varySpeedPenaltyWithAndWithoutEps(x[i], x[i+1], \
                x[imageHeight+i], x[imageHeight+i+1], 2)

        else:
            speedPenaltyFwd = 0

        gradientArray.append((rWidthPenalty + accelPenaltyBwd + accelPenaltyHere + \
            accelPenaltyFwd + speedPenaltyBwd + speedPenaltyFwd)/EPS)

    return np.array(gradientArray)

#for i in penaltyFuncPrime(x0):
#    print i

#print np.linalg.norm(penaltyFuncPrime(x0))

x = x0
for i in range(1000):
    grad = penaltyFuncPrime(x)

    x -= grad*learnRate
    averageSpeed = computeAverageSpeed(x)

    print i, "/ 1000 ||", np.linalg.norm(grad)

thetas = x[:imageHeight]
rs = x[imageHeight:]

prism = cm.get_cmap(name="gist_rainbow")
norm = Normalize(0, imageHeight)
scalarMappable = cm.ScalarMappable(norm=norm, cmap=prism)

counter = 0

for theta, r in zip(thetas, rs):
    currentX = r*cos(theta)
    currentY = r*sin(theta)

    p.plot(currentX, currentY, marker="o", color=scalarMappable.to_rgba(counter))
    counter += 1

p.savefig("tracking_result.png")
p.show()

#for i in x:
#    print i

#p.plot(x[:imageHeight])
#p.show()

#p.clf()
#p.plot(x[imageHeight:])
#p.show()

#print fmin_bfgs(penaltyFunc, x0, retall=True, fprime=penaltyFuncPrime, \
#    callback = computeAverageSpeed, gtol=0)
