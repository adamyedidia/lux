from __future__ import division
import numpy as np
import pylab
from math import pi, sqrt
import matplotlib.pyplot as p
from PIL import Image
import pickle

EPS = 1e-9

showExample = False
test = True

def lightResponse(lightX, viewX, deltaX):
    if lightX > viewX + EPS:
        return 0
    else:
        return deltaX/(pi * sqrt(1 + (lightX + viewX)**2)**3)

def createGarbleMatrix(sideLength, maxX):
    deltaX = maxX/sideLength
    responseList = []

    for lightXLarge in range(sideLength):
        lightX = maxX * lightXLarge / sideLength
        responseList.append([])
        for viewXLarge in range(sideLength):
            viewX = maxX * viewXLarge / sideLength
            responseList[-1].append(lightResponse(lightX, viewX, deltaX))

    return np.array(responseList)

def getQ(H, maxDist):
    u,s,v = np.linalg.svd(H)
    return u, np.diag(s), v

def createGarbleMatrixX(imageMat, maxX):
    return createGarbleMatrix(imageMat.shape[1], maxX)

def createGarbleMatrixY(imageMat, maxY):
    return createGarbleMatrix(imageMat.shape[0], maxY)

def garbleImageY(imageMat, garbleMatrixY):
    return np.dot(garbleMatrixY, imageMat)

def garbleImageX(imageMat, garbleMatrixX):
    return np.transpose(np.dot(garbleMatrixX, np.transpose(imageMat)))

def ungarbleImageXOld(imageMat, garbleMatrixX, sigma):
    return np.transpose(np.dot(np.linalg.inv(garbleMatrixX), np.transpose(imageMat)))

def ungarbleImageX(imageMat, garbleMatrixX, sigma):
    sideLength = garbleMatrixX.shape[0]

#    print "garbleX", garbleMatrixX

#    print "H^T H", np.linalg.det(np.dot(np.transpose(garbleMatrixX), \
#        garbleMatrixX) + sigma*sigma*np.identity(sideLength))

#    print "H^T H inverse", np.linalg.inv(np.dot(np.transpose(garbleMatrixX), \
#        garbleMatrixX) + sigma*sigma*np.identity(sideLength))

#    print "H inverse", np.linalg.inv(garbleMatrixX)

    # (A^T A + sigma**2 I)^-1 A^T
    smoothInverter = np.dot(np.linalg.inv(np.dot(np.transpose(garbleMatrixX), \
        garbleMatrixX) + sigma*sigma*np.identity(sideLength)), \
        np.transpose(garbleMatrixX))

#    print "smooth inverter", smoothInverter

    return np.transpose(np.dot(smoothInverter, np.transpose(imageMat)))

def ungarbleImageYOld(imageMat, garbleMatrixX, sigma):
    return np.dot(np.linalg.inv(garbleMatrixX), imageMat)

def ungarbleImageY(imageMat, garbleMatrixY, sigma):
    sideLength = garbleMatrixY.shape[0]

    print np.linalg.inv(np.dot(np.transpose(garbleMatrixY), \
        garbleMatrixY) + sigma*sigma*np.identity(sideLength))

    # (A^T A + sigma**2 I)^-1 A^T
    smoothInverter = np.dot(np.linalg.inv(np.dot(np.transpose(garbleMatrixY), \
        garbleMatrixY) + sigma*sigma*np.identity(sideLength)), \
        np.transpose(garbleMatrixY))

    return np.dot(smoothInverter, imageMat)

if showExample:

    imRaw = Image.open("adam_h.jpeg")
    im = np.array(imRaw).astype(float)

    print im.shape
    print np.swapaxes(np.swapaxes(im, 0, 2), 1, 2).shape

    rearrangedIm = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
    imageRed = rearrangedIm[0]
    imageGreen = rearrangedIm[1]
    imageBlue = rearrangedIm[2]
    matShape = imageRed.shape

    gmx = createGarbleMatrixX(imageRed, 2)
    gmy = createGarbleMatrixY(imageRed, matShape[0]/matShape[1])

    gmxff = createGarbleMatrixX(imageRed, 1e-6)
    gmyff = createGarbleMatrixY(imageRed, 1e-6)

    zeroBlue = np.zeros(matShape)
    zeroGreen = np.zeros(matShape)

    garbledImageRed = garbleImageY(garbleImageX(imageRed, gmx), gmy)
    garbledImageGreen = garbleImageY(garbleImageX(imageGreen, gmx), gmy)
    garbledImageBlue = garbleImageY(garbleImageX(imageBlue, gmx), gmy)

    rearrangedGarbledImage = np.array([garbledImageRed, garbledImageGreen, garbledImageBlue])

    garbledImage = np.swapaxes(np.swapaxes(rearrangedGarbledImage,1,2),0,2)
    boundedGarbledImage = np.minimum(np.maximum(garbledImage, np.zeros(im.shape)), \
        np.full(shape=im.shape, fill_value=255))

    pickle.dump(boundedGarbledImage, open("garbled.p", "w"))

    pylab.imshow((40*boundedGarbledImage).astype(np.uint8))
    p.show()

    print garbledImageRed

    noisyRed = garbledImageRed + np.random.normal(size=matShape,scale=1e-5)
    noisyGreen = garbledImageGreen + np.random.normal(size=matShape,scale=1e-5)
    noisyBlue = garbledImageBlue + np.random.normal(size=matShape,scale=1e-5)

    ungarbledImageRed = ungarbleImageY(ungarbleImageX(noisyRed, gmx), gmy)
    ungarbledImageGreen = ungarbleImageY(ungarbleImageX(noisyGreen, gmx), gmy)
    ungarbledImageBlue = ungarbleImageY(ungarbleImageX(noisyBlue, gmx), gmy)

    #ungarbledImageRed = ungarbleImageY(ungarbleImageX(noisyRed, gmxff), gmyff)
    #ungarbledImageGreen = ungarbleImageY(ungarbleImageX(noisyGreen, gmxff), gmyff)
    #ungarbledImageBlue = ungarbleImageY(ungarbleImageX(noisyBlue, gmxff), gmyff)

    print ungarbledImageBlue

    rearrangedReddifiedImage = np.array([ungarbledImageRed, ungarbledImageGreen, ungarbledImageBlue])
    reddifiedImage = np.swapaxes(np.swapaxes(rearrangedReddifiedImage, 1, 2), 0, 2)*1e-12*0.5


    print rearrangedReddifiedImage.shape
    print reddifiedImage.shape

    maxZeroMatrix = np.maximum(reddifiedImage, np.zeros(im.shape))

    print maxZeroMatrix

    boundedMatrix = np.minimum(maxZeroMatrix, \
        np.full(shape=im.shape, fill_value=255))

    print boundedMatrix

    pylab.imshow((boundedMatrix).astype(np.uint8))
    #pylab.imshow(im.astype(np.uint8))
    p.show()

if test:
    print getQ(createGarbleMatrix(5, 1), 1)

#arr = []
#for i in range(100):
#    arr.append([])
#    for j in range(100):
#        if abs(i-j) <= 1:
#            arr[-1].append(0)
#        else:
#            arr[-1].append(1)


#garbleMat = createGarbleMatrix(100, 1e-6)

#invGarbleMat = np.linalg.inv(garbleMat)
#print np.multiply(invGarbleMat, np.array(arr))[0]

#p.matshow(np.multiply(invGarbleMat, np.array(arr)))
#p.matshow(np.linalg.inv(garbleMat))
#p.show()
