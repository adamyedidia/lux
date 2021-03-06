from math import sqrt
import pylab
import imageio
import numpy as np
from math import acos, pi, cos, sin, log, floor, ceil, atan2, tan
import matplotlib.pyplot as p
import matplotlib.cm as cm
import matplotlib.patches as ptch
from matplotlib.colors import Normalize
import pickle
import sys
from PIL import Image
from PIL import ImageFilter
from video_magnifier import turnVideoIntoListOfFrames, viewFrame
import os
import string
import pickle
from scipy.signal import convolve2d, medfilt
from scipy.ndimage.filters import gaussian_filter
import sys
from scipy.optimize import fmin_tnc, brute
import random
import time
from image_distortion_simulator import doFuncToEachChannel, imageify
from import_1dify import fuzzyLookup2D
from sklearn.neighbors import BallTree

test = False
flag = False
raw = False
viewDiff = False
downSample = False
downSample2 = False
rawWithSubtract = False
rawWithBlur = False
batchMovie = False
batchSmallerMovie = False
batchRickMortyMovie = False
batchMulanMovie = False
batchOfficeMovie = False
processBlindDeconvVideo = False
downsampleWinnie = False
weirdAngle = False
weirdAngleSim = False
weirdAngleSimMovie = False
weirdAngleSimRecovery = False
processDualVideo = False
macarena = False
video36225 = False
orange = False
bld66 = False
bld34 = False
stata = False
fan = False
fan_fine = False
fan_monitor = False
plant = False
plant_fine = False
plant_monitor = False
glass_rose = False
glass_rose_2 = False
matthew_wall = False
impulse_movie = False
darpa_vid = False
darpa_gt = False
obama = False
darpa_fan = False
darpa_iphone_fan = False
darpa_mannequin = False
particle_vid = False
sidewalk = False
lion_king_bottle = False
lion_king_bottle_sparsified = False
lion_king_bottle_build_array = True
colorStackTest = False

def actOnRGB(rgbArray, func):
    rearrangedIm = np.swapaxes(np.swapaxes(rgbArray, 0, 2), 1, 2)
    imageRed = rearrangedIm[0]
    imageGreen = rearrangedIm[1]
    imageBlue = rearrangedIm[2]

    imageRedProcessed = func(imageRed)
    imageGreenProcessed = func(imageGreen)
    imageBlueProcessed = func(imageBlue)

    imageProcessed = np.swapaxes(np.swapaxes(np.array([imageRedProcessed, \
        imageGreenProcessed, imageBlueProcessed]), 1, 2), 0, 2)

    return imageProcessed

def addNoise(arr, noiseSigma):    
    return arr + np.random.normal(loc=0, scale=noiseSigma, size=arr.shape)

def blur2DImage(arr, blurRadius):
#    print arr.shape

    rearrangedIm = np.swapaxes(np.swapaxes(arr, 0, 2), 1, 2)
    imageRed = rearrangedIm[0]
    imageGreen = rearrangedIm[1]
    imageBlue = rearrangedIm[2]

    imageRed = gaussian_filter(imageRed, blurRadius, truncate=4.)
    imageGreen = gaussian_filter(imageGreen, blurRadius, truncate=4.)
    imageBlue = gaussian_filter(imageBlue, blurRadius, truncate=4.)

    blurredImage = np.swapaxes(np.swapaxes(np.array([imageRed, imageGreen, \
        imageBlue]), 1, 2), 0, 2)

    return blurredImage

def medfiltIm(arr):
    rearrangedIm = np.swapaxes(np.swapaxes(arr, 0, 2), 1, 2)
    imageRed = rearrangedIm[0]
    imageGreen = rearrangedIm[1]
    imageBlue = rearrangedIm[2]

    imageRed = medfilt(imageRed)
    imageGreen = medfilt(imageGreen)
    imageBlue = medfilt(imageBlue)

    medfiltedImage = np.swapaxes(np.swapaxes(np.array([imageRed, imageGreen, \
        imageBlue]), 1, 2), 0, 2)

    return medfiltedImage


def padIntegerWithZeros(x, maxLength):
    if x == 0:
        return "0"*maxLength

    assert log(x, 10) < maxLength

    return "0"*(maxLength-int(floor(log(x, 10)))-1) + str(x)

def turnVideoIntoArray(vid, firstFrame=0, lastFrame=None):
    return np.array(turnVideoIntoListOfFrames(vid, firstFrame, lastFrame))

def batchArrayAlongZeroAxis(arr, batchSize):

    listOfBigFrames = []
    frameSum = np.zeros(arr[0].shape)
    numFrames = len(arr)

    for i in range(numFrames):
        frameSum += arr[i]

        if i % batchSize == batchSize - 1:
            listOfBigFrames.append(frameSum / batchSize)
            frameSum = np.zeros(arr[0].shape)

    if numFrames % batchSize != 0:
        listOfBigFrames.append(frameSum / (numFrames % batchSize))

#    print len(listOfBigFrames)
    return np.array(listOfBigFrames)

def batchArrayAlongAxis(arr, axis, batchSize):
    rearrangedArr = np.swapaxes(arr, 0, axis)
    batchedArray = batchArrayAlongZeroAxis(rearrangedArr, batchSize)
    return(np.swapaxes(batchedArray, 0, axis))

def convertTimeToSeconds(timeString):
#    colonIndex = string.find(timeString, ":")
    colonIndex = timeString.find(":")
    minutes = int(timeString[:colonIndex])
    seconds = int(timeString[colonIndex+1:])

    return 60*minutes + seconds

def batchAndDifferentiate(arr, listOfResponses):
    dim  = len(arr.shape)

    assert dim == len(listOfResponses)

    # batch things
#    print "Batching..."
    for i in range(dim):
        arr = batchArrayAlongAxis(arr, i, listOfResponses[i][0])

#    viewFrame(arr, 1e0, False)

    # take gradients
#    print "Differentiating..."
    for i in range(dim - 1, -1, -1):
        if listOfResponses[i][1]:
            arr = np.gradient(arr, axis=i)

#            viewFrame(arr, 3e2, True)

#    arr = blur2DImage(arr, 5)

#    viewFrame(arr, 3e2, True)

    return arr

def batchAndIntegrate(arr, listOfResponses):
    dim  = len(arr.shape)

    assert dim == len(listOfResponses)

    # batch things
#    print "Batching..."
    for i in range(dim):
        arr = batchArrayAlongAxis(arr, i, listOfResponses[i][0])

#    viewFrame(arr, 1e0, False)

    # take gradients
#    print "Differentiating..."
    for i in range(dim - 1, -1, -1):
        if listOfResponses[i][1]:
            arr = np.cumsum(arr, axis=i)

#            viewFrame(arr, 3e2, True)

#    arr = blur2DImage(arr, 5)

#    viewFrame(arr, 3e2, True)

    return arr



def convertArrayToVideo(arr, magnification, filename, frameRate, \
    adaptiveScaling=True, differenceImage=True, verbose=False):

    assert len(arr.shape) == 4

    print(arr.shape)
    numFrames = arr.shape[0]
    logNumFrames = int(floor(log(numFrames, 10)))+1
    print("logNumFrames", logNumFrames)

    os.system("rm video_trash/" + filename + "_*.png")

    for i, frame in enumerate(arr):
#        print frame.shape
 #       print type(frame[0][0][0])

        if verbose:
            print(i)

        p.clf()
        p.axis("off")
        viewFrame(frame, magnification=magnification, filename="video_trash/" + filename + "_" + \
            padIntegerWithZeros(i, logNumFrames) + ".png", differenceImage=differenceImage, \
            adaptiveScaling=adaptiveScaling)

    numDigits = ceil(log(len(arr), 10))

    os.system("ffmpeg -r " + str(frameRate) + " -f image2 -s 500x500 " + \
        "-i video_trash/" + filename + "_%0" + str(int(numDigits)) + "d.png " + \
        "-vcodec libx264 -crf 25 -pix_fmt yuv420p " + filename + ".mp4")
    os.system("y")

def convertTwoArraysToVideo(arr1, arr2, magnification, filename, frameRate, \
    adaptiveScaling=True, differenceImage=True, verbose=False):
    assert len(arr1.shape) == 4
    assert len(arr2.shape) == 4

    numFrames = arr1.shape[0]
    logNumFrames = int(floor(log(numFrames, 10)))+1

    os.system("rm video_trash/" + filename + "_*.png")

    for i, frameTuple in enumerate(zip(arr1, arr2)):
        frame1, frame2 = frameTuple

        if verbose:
            print(i)

        p.clf()
        p.subplot(211)
        p.axis("off")
        viewFrame(frame1, magnification=magnification, filename="pass", differenceImage=differenceImage, \
            adaptiveScaling=adaptiveScaling, relax=True)

        p.subplot(212)
        p.axis("off")
        viewFrame(frame2, magnification=magnification, filename="pass", differenceImage=differenceImage, \
            adaptiveScaling=adaptiveScaling, relax=True)

        p.savefig("video_trash/" + filename + "_" + \
            padIntegerWithZeros(i, logNumFrames) + ".png")

    numDigits = ceil(log(len(arr1), 10))

    os.system("ffmpeg -r " + str(frameRate) + " -f image2 -s 500x500 " + \
        "-i video_trash/" + filename + "_%0" + str(int(numDigits)) + "d.png " + \
        "-vcodec libx264 -crf 25 -pix_fmt yuv420p " + filename + ".mp4")
    os.system("y")

def convertFourArraysToVideo(arr1, arr2, arr3, arr4, magnification, filename, \
    frameRate, adaptiveScaling=True, differenceImage=True):

    assert len(arr1.shape) == 4
    assert len(arr2.shape) == 4
    assert len(arr3.shape) == 4
    assert len(arr4.shape) == 4

    numFrames = arr1.shape[0]
    logNumFrames = int(floor(log(numFrames, 10)))+1

    os.system("rm video_trash/" + filename + "_*.png")

    for i, frameTuple in enumerate(zip(arr1, arr2, arr3, arr4)):
        frame1, frame2, frame3, frame4 = frameTuple

        print(i)

        p.clf()
        p.subplot(221)
        p.axis("off")
        viewFrame(frame1, magnification=magnification, filename="pass", differenceImage=differenceImage, \
            adaptiveScaling=adaptiveScaling, relax=True)

        p.subplot(222)
        p.axis("off")
        viewFrame(frame2, magnification=magnification, filename="pass", differenceImage=differenceImage, \
            adaptiveScaling=adaptiveScaling, relax=True)

        p.subplot(223)
        p.axis("off")
        viewFrame(frame3, magnification=magnification, filename="pass", differenceImage=differenceImage, \
            adaptiveScaling=adaptiveScaling, relax=True)

        p.subplot(224)
        p.axis("off")
        viewFrame(frame4, magnification=magnification, filename="pass", differenceImage=differenceImage, \
            adaptiveScaling=adaptiveScaling, relax=True)

        p.savefig("video_trash/" + filename + "_" + \
            padIntegerWithZeros(i, logNumFrames) + ".png")

    numDigits = ceil(log(len(arr1), 10))

    os.system("ffmpeg -r " + str(frameRate) + " -f image2 -s 500x500 " + \
        "-i video_trash/" + filename + "_%0" + str(int(numDigits)) + "d.png " + \
        "-vcodec libx264 -crf 25 -pix_fmt yuv420p " + filename + ".mp4")
    os.system("y")



def getFrameAtTime(frameTime, videoTime, numFrames):
    return int(convertTimeToSeconds(frameTime) / \
        convertTimeToSeconds(videoTime) * numFrames)

def processVideo(vid, vidLength, listOfResponses, filename, magnification=1, \
    firstFrame=0, lastFrame=None, toVideo=False):

    arr = turnVideoIntoArray(vid, firstFrame, lastFrame)
    arr = batchAndDifferentiate(arr, listOfResponses)

    print(arr.shape)

    numFramesInOriginalVideo = len(vid)
    originalFrameRate = numFramesInOriginalVideo / convertTimeToSeconds(vidLength)

    newFrameRate = originalFrameRate / listOfResponses[0][0]

    if toVideo:
        convertArrayToVideo(arr, magnification, filename, newFrameRate)
    else:
        pickle.dump(arr, open(filename + ".p", "w"))

def processVideoCheap(vid, vidLength, listOfResponses, filename, magnification=1,
    firstFrame=0, lastFrame=None, toVideo=False, minY=None, maxY=None, minX=None, 
    maxX=None, returnSomething=False):

    listOfBatchedFrames = []

    print(firstFrame, lastFrame)

    for i in range(firstFrame, lastFrame):
        print(i-firstFrame, "/", lastFrame - firstFrame)

#        try:
        im = vid.get_data(i)
#        except:
#            im = vid[i]
        frame = np.array(im).astype(float)

        if minY == None and minX == None:
            batchedFrame = batchAndDifferentiate(frame, listOfResponses[1:])
        elif minX == None:
            batchedFrame = batchAndDifferentiate(frame, listOfResponses[1:])
            batchedFrame = batchedFrame[minY:maxY,:]
        elif minY == None:
            batchedFrame = batchAndDifferentiate(frame, listOfResponses[1:])
            batchedFrame = batchedFrame[:,minX:maxX]
        else:
            batchedFrame = batchAndDifferentiate(frame, listOfResponses[1:])

#            if i % 100 == 0:
#                print batchedFrame.shape
#                viewFrame(batchedFrame)


            batchedFrame = batchedFrame[minY:maxY,minX:maxX]            
        if i == firstFrame + 100:
            print(batchedFrame.shape)

            viewFrame(batchedFrame)

#        if i == 100:

        listOfBatchedFrames.append(batchedFrame)

#    print listOfResponses

    arr = batchAndDifferentiate(np.array(listOfBatchedFrames), \
        [listOfResponses[0]] + [(1, False), (1, False), (1, False)])

    numFramesInOriginalVideo = len(vid)
    originalFrameRate = numFramesInOriginalVideo / convertTimeToSeconds(vidLength)

    newFrameRate = originalFrameRate / listOfResponses[0][0]       
    if returnSomething:
        return arr
    else:
        if toVideo:
            convertArrayToVideo(arr, magnification, filename, newFrameRate)
        else:
            pickle.dump(arr, open(filename + ".p", "wb"))

def processVideoCheapWithPixels(vid, listOfPixels, vidLength, listOfResponses, filename, magnification=1,
    firstFrame=0, lastFrame=None, toVideo=False, minY=None, maxY=None, minX=None, 
    maxX=None, returnSomething=False):

    listOfBatchedFrames = []

    print(firstFrame, lastFrame)

    for i in range(firstFrame, lastFrame):
        print(i-firstFrame, "/", lastFrame - firstFrame)



#        try:
        im = vid.get_data(i)
#        except:
#            im = vid[i]
        frame = np.array(im).astype(float)

#        for pixel in listOfPixels[:10000]:

#            if random.random() < 0.01:
#                p.plot(pixel[1]+minX, pixel[0]+minY, "bo")
#        viewFrame(frame, relax=True)

        trueFrame = []
        for pixel in listOfPixels:
#            print(pixel)
#            print(frame.shape, (pixel[0]+minY, pixel[1]+minX))
            trueFrame.append(frame[pixel[0]+minY][pixel[1]+minX])

        if minY == None and minX == None:
            batchedFrame = batchAndDifferentiate(frame, listOfResponses[1:])
        elif minX == None:
            batchedFrame = batchAndDifferentiate(frame, listOfResponses[1:])
            batchedFrame = batchedFrame[minY:maxY,:]
        elif minY == None:
            batchedFrame = batchAndDifferentiate(frame, listOfResponses[1:])
            batchedFrame = batchedFrame[:,minX:maxX]
        else:
            batchedFrame = batchAndDifferentiate(frame, listOfResponses[1:])

#            if i % 100 == 0:
#                print batchedFrame.shape
#                viewFrame(batchedFrame)


            batchedFrame = batchedFrame[minY:maxY,minX:maxX]            
        if i == firstFrame + 100:
            print(batchedFrame.shape)

            viewFrame(np.reshape(np.array(trueFrame), (100, 100, 3)))

#        if i == 100:

        listOfBatchedFrames.append(np.reshape(np.array(trueFrame), (100, 100, 3)))

#    print listOfResponses

    arr = batchAndDifferentiate(np.array(listOfBatchedFrames), \
        [listOfResponses[0]] + [(1, False), (1, False), (1, False)])

    numFramesInOriginalVideo = len(vid)
    originalFrameRate = numFramesInOriginalVideo / convertTimeToSeconds(vidLength)

    newFrameRate = originalFrameRate / listOfResponses[0][0]       
    if returnSomething:
        return arr
    else:
        if toVideo:
            convertArrayToVideo(arr, magnification, filename, newFrameRate)
        else:
            pickle.dump(arr, open(filename + ".p", "wb"))


def getTheta(warpPoint, x, y):
    deltaX = x - warpPoint[0]
    deltaY = y - warpPoint[1]

#    print warpPoint, deltaX, deltaY
#    print 

    return atan2(deltaY, deltaX)

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def getXY(warpPoint1, warpPoint2, theta1, theta2):
    x00 = warpPoint1[0]
    y00 = warpPoint1[1]
    x01 = warpPoint1[0] + cos(theta1)
    y01 = warpPoint1[1] + sin(theta1)

    x10 = warpPoint2[0]
    y10 = warpPoint2[1]
    x11 = warpPoint2[0] + cos(theta2)
    y11 = warpPoint2[1] + sin(theta2)    

    if x01-x00 == 0:
        m0 = sign(y01-y00)*1e4
    else:
        m0 = (y01-y00)/(x01-x00)

    if x11 - x10 == 0:
        m1 = sign(y11-y10)*1e4
    else:
        m1 = (y11-y10)/(x11-x10)

    b0 = y00 - m0*x00
    b1 = y10 - m1*x10

#    print "Ms", m0, m1
#    print "Bs", b0, b1

    if m1-m0 == 0:
        x = -sign(b1-b0)*1e4        
    else:
        x = -(b1-b0)/(m1-m0)
    
    y = m0*x + b0

#    print "a", x00, y00, x01, y01
#    print "b", x10, y10, x11, y11
#    print "c", b0, b1
#    print "d", m0, m1

#    print "e", theta1, theta2, x, y

    return x, y

def getMinAndMaxTheta(warpPoint, arrShape):

    if warpPoint[0] > 0 and warpPoint[0] < arrShape[0]-1 and \
        warpPoint[1] > 0 and warpPoint[1] < arrShape[1]-1:

        minTheta = -pi
        maxTheta = pi

        return minTheta, maxTheta

    elif warpPoint[0] > arrShape[0] - 1 and \
        warpPoint[1] > 0 and warpPoint[1] < arrShape[1]-1:

        minTheta = pi
        maxTheta = -pi        

        for x in [0, arrShape[0]-1]:
            for y in [0, arrShape[1]-1]:
                theta = getTheta(warpPoint, x, y)

                if theta < 0:
                    theta += 2*pi

                minTheta = min(theta, minTheta)
                maxTheta = max(theta, maxTheta)

        return minTheta, maxTheta

    else:

        minTheta = pi
        maxTheta = -pi        

        for x in [0, arrShape[0]-1]:
            for y in [0, arrShape[1]-1]:
                theta = getTheta(warpPoint, x, y)

                minTheta = min(theta, minTheta)
                maxTheta = max(theta, maxTheta)

        return minTheta, maxTheta

def initializeThetaArray(minTheta1, maxTheta1, minTheta2, maxTheta2, numSteps):
    returnArray = np.zeros((numSteps, numSteps, 3))
    counterArray = np.zeros((numSteps, numSteps, 3))


    def addToArray(rgb, theta1, theta2):

#        print (theta1-minTheta1)/(maxTheta1-minTheta1+1e-4)*numSteps, (theta2-minTheta2)/(maxTheta2-minTheta2+1e-4)*numSteps 

        # special case for the -pi pi meeting point when the warpPoint is in the eastern octant
        if maxTheta1 > pi and theta1 < 0:
            theta1 += 2*pi

        if maxTheta2 > pi and theta2 < 0:
            theta2 += 2*pi

#        print theta1, theta2

        index1 = int(floor((theta1-minTheta1)/(maxTheta1-minTheta1+1e-4)*numSteps))
        index2 = int(floor((theta2-minTheta2)/(maxTheta2-minTheta2+1e-4)*numSteps))

#        print "adding", rgb, "to (", index1, ",", index2, ")"

#        print index1, index2

        returnArray[index1][index2] += rgb
        counterArray[index1][index2] += np.array([1,1,1])

    return returnArray, counterArray, addToArray

def divWithZero(x, y):
    if x == 0:
        return 0
    else:
        return x/y

def getTotalArrayIntensity(arr):
    rawTotalIntensity = np.sum(np.sum(np.sum(np.abs(arr), 0), 0), 0)
    numNonzeroEntries = np.sum(np.sum(np.sum(np.vectorize(lambda x: 1*(x!=0))(arr), 0), 0), 0)

    return rawTotalIntensity #/ numNonzeroEntries

divWithZeroVectorized = np.vectorize(divWithZero)

def warp2DArrayOld(arr, warpPoint1, warpPoint2, numSteps=100):
    arrShape = arr.shape

    minTheta1, maxTheta1 = getMinAndMaxTheta(warpPoint1, arrShape)
    minTheta2, maxTheta2 = getMinAndMaxTheta(warpPoint2, arrShape)

#    print minTheta1, maxTheta1
#    print minTheta2, maxTheta2

#    print "warpPoint1", warpPoint1
#    print "warpPoint2", warpPoint2

    thetaArray, counterArray, addToArray = initializeThetaArray(minTheta1, maxTheta1, minTheta2, maxTheta2, numSteps)

    for x, xVal in enumerate(arr):
        for y, yVal in enumerate(xVal):
            theta1 = getTheta(warpPoint1, x, y)
            theta2 = getTheta(warpPoint2, x, y)

#            print "theta1", theta1, "theta2", theta2

            addToArray(yVal, theta1, theta2)

#    print thetaArray, counterArray

    thetaArray = divWithZeroVectorized(thetaArray, counterArray)

#    print thetaArray

    return thetaArray

def warp2DArray(arr, warpPoint1, warpPoint2, pixellationFactor=1, numSamples=1):
    arrShape = arr.shape

    minTheta1, maxTheta1 = getMinAndMaxTheta(warpPoint1, arrShape)
    minTheta2, maxTheta2 = getMinAndMaxTheta(warpPoint2, arrShape)

    xs = np.linspace(0, arrShape[0], int(arrShape[0]/pixellationFactor))
    ys = np.linspace(0, arrShape[1], int(arrShape[1]/pixellationFactor))

    returnArray = []

    for i, x in enumerate(xs[:-1]):
        returnArray.append([])

        for j, y in enumerate(ys[:-1]):

            if numSamples == 1:
                theta1 = x/arrShape[0] * (maxTheta1-minTheta1) + minTheta1
                theta2 = y/arrShape[1] * (maxTheta2-minTheta2) + minTheta2

    #            print minTheta1, maxTheta1, minTheta2, maxTheta2
                val = thetaLookup(arr, theta1, warpPoint1, theta2, warpPoint2)

                returnArray[-1].append(val)

            else:
                averageVal = np.array([0.,0.,0.])
                for subX in np.linspace(xs[i], xs[i+1], numSamples):
                    for subY in np.linspace(ys[j], ys[j+1], numSamples):
                        theta1 = subX/arrShape[0] * (maxTheta1-minTheta1) + minTheta1
                        theta2 = subY/arrShape[1] * (maxTheta2-minTheta2) + minTheta2

                        averageVal += thetaLookup(arr, theta1, warpPoint1, theta2, warpPoint2).astype(float)

                averageVal = averageVal / float(numSamples*numSamples)

                returnArray[-1].append(averageVal)    

    return np.array(returnArray)

def initializeXYArray(minX, maxX, minY, maxY, xSteps, ySteps):
    returnArray = np.zeros((xSteps, ySteps, 3))
    counterArray = np.zeros((xSteps, ySteps, 3))

    def addToArray(rgb, x, y):

#        print (theta1-minTheta1)/(maxTheta1-minTheta1+1e-4)*numSteps, (theta2-minTheta2)/(maxTheta2-minTheta2+1e-4)*numSteps 

    # special case for the -pi pi meeting point when the warpPoint is in the eastern octant
#    if maxTheta1 > pi and theta1 < 0:
#        theta1 += 2*pi

#    if maxTheta2 > pi and theta2 < 0:
#        theta2 += 2*pi

#        print theta1, theta2
        try:

            print(x, y)

            index1 = int(floor((x-minX)/(maxX-minX+1e-4)*xSteps))
            index2 = int(floor((y-maxY)/(maxY-minY+1e-4)*ySteps))

    #        print "adding", rgb, "to (", index1, ",", index2, ")"

    #        print index1, index2

            if index1 >= 0 and index2 >= 0:
                returnArray[index1][index2] += rgb
                counterArray[index1][index2] += np.array([1,1,1])
        except:
            pass

    return returnArray, counterArray, addToArray


def preWarpArrayOld(arr, warpPoint1, warpPoint2, numSteps=100):
    arrShape = arr.shape

    returnArray, counterArray, addToArray = initializeXYArray(-50, 50, 0, 99, 13, 20)

    for i, xVal in enumerate(arr):
        for j, yVal in enumerate(xVal):
            theta1 = i*2*pi/arrShape[0]
            theta2 = j*2*pi/arrShape[1]

            x, y = getXY(warpPoint1, warpPoint2, theta1, theta2)

            addToArray(yVal, x, y)

    returnArray = divWithZeroVectorized(returnArray, counterArray)

    return returnArray

def thetaReverseLookup(arr, theta1, minTheta1, maxTheta1, theta2, minTheta2, maxTheta2):
    arrShape = arr.shape

    if maxTheta1 > pi and theta1 < 0:
        theta1 += 2*pi

    if maxTheta2 > pi and theta2 < 0:
        theta2 += 2*pi

    theta1Fraction = (theta1 - minTheta1)/(maxTheta1 - minTheta1)
    theta2Fraction = (theta2 - minTheta2)/(maxTheta2 - minTheta2)

    xCor = (arrShape[0]-2.5)*theta1Fraction
    yCor = (arrShape[1]-2.5)*theta2Fraction

#    print xCor

    return fuzzyLookup2D(arr, xCor, yCor)

def thetaLookup(arr, theta1, warpPoint1, theta2, warpPoint2):
    arrShape = arr.shape

    x, y = getXY(warpPoint1, warpPoint2, theta1, theta2)

#    print warpPoint1, warpPoint2

#    print theta1, theta2

#    print x, y

    if x < 0 or x >= arrShape[0]-1 or y < 0 or y >= arrShape[1]-1:

        return np.array([0, 0, 0])

    return fuzzyLookup2D(arr, x, y)

def preWarpArray(arr, warpPoint1, warpPoint2, numSamples=1):
    arrShape = arr.shape

    minTheta1, maxTheta1 = getMinAndMaxTheta(warpPoint1, arrShape)
    minTheta2, maxTheta2 = getMinAndMaxTheta(warpPoint2, arrShape)

    xs = np.linspace(0, arrShape[0], arrShape[0])
    ys = np.linspace(0, arrShape[1], arrShape[1])

    returnArray = []

    for i, x in enumerate(xs[:-1]):
        returnArray.append([])

        for j, y in enumerate(ys[:-1]):

            if numSamples == 1:
                theta1 = getTheta(warpPoint1, x, y)
                theta2 = getTheta(warpPoint2, x, y)

                val = thetaReverseLookup(arr, theta1, minTheta1, maxTheta1, theta2, minTheta2, maxTheta2)

                returnArray[-1].append(val)

            else:
                averageVal = 0

                for subX in np.linspace(xs[i], xs[i+1], numSamples):
                    for subY in np.linspace(ys[j], ys[j+1], numSamples):
                        theta1 = getTheta(warpPoint1, subX, subY)
                        theta2 = getTheta(warpPoint2, subX, subY)

                        val = thetaReverseLookup(arr, theta1, minTheta1, maxTheta1, theta2, minTheta2, maxTheta2)

                        averageVal += val

                averageVal /= numSamples*numSamples

                returnArray[-1].append(averageVal)

    return np.array(returnArray)

def getWarpPointFromAngles(arrShape, theta, phi):
    return np.array([arrShape[0]*cos(theta)*tan(phi), arrShape[1]*sin(theta)*tan(phi)]) + \
        np.array([arrShape[0]/2, arrShape[1]/2])

def getWarpPointFromAnglesVariableCenter(arrShape, theta, phi, x, y):
    return np.array([arrShape[0]*cos(theta)*tan(phi), arrShape[1]*sin(theta)*tan(phi)]) + \
        np.array([x*arrShape[0], y*arrShape[1]])

def quadraticPenaltyForOutsideTanRange(phi1, phi2):
    penalty = 0
    EPS = 1e-5

    if phi1 > pi/2-EPS:
        penalty += (phi1 - (pi/2-EPS))**2
        phi1 = pi/2-EPS

    if phi1 < -(pi/2-EPS):
        penalty += (phi1 + (pi/2-EPS))**2
        phi1 = -(pi/2-EPS)

    if phi2 > pi/2-EPS:
        penalty += (phi2 - (pi/2-EPS))**2
        phi2 = pi/2-EPS

    if phi2 < -(pi/2-EPS):
        penalty += (phi2 + (pi/2-EPS))**2
        phi2 = -(pi/2-EPS)

    return penalty, phi1, phi2

def evaluateAngleMaker(arr, rightAngleOnly=True):
    arrShape = arr.shape

    def evaluateAngle(angleArray):
        if rightAngleOnly:
            theta1 = angleArray[0]
            phi1 = angleArray[1]
            theta2 = theta1 + pi/2
            phi2 = angleArray[2]            

        else:
            theta1 = angleArray[0]
            phi1 = angleArray[1]
            theta2 = angleArray[2]
            phi2 = angleArray[3]

#        print "current angles", theta1, phi1, theta2, phi2

        penalty, phi1, phi2 = quadraticPenaltyForOutsideTanRange(phi1, phi2)

        warpPoint1 = getWarpPointFromAnglesVariableCenter(arrShape, theta1, phi1)
        warpPoint2 = getWarpPointFromAnglesVariableCenter(arrShape, theta2, phi2)

        warpedArray = warp2DArray(arr, warpPoint1, warpPoint2, numSteps=20)

        diffArray = batchAndDifferentiate(warpedArray, [(1, True), (1, True), (1, False)])

        totalArrayIntensity = getTotalArrayIntensity(diffArray)
#        print "intensity", totalArrayIntensity

        return totalArrayIntensity + penalty

    return evaluateAngle

def evaluateAngleMakerVariableCenter(arr, rightAngleOnly=True):
    arrShape = arr.shape

    def evaluateAngle(angleArray):
        if rightAngleOnly:
            theta1 = angleArray[0]
            phi1 = angleArray[1]
            theta2 = theta1 + pi/2
            phi2 = angleArray[2]      
            x = angleArray[3]
            y = angleArray[4]      

        else:
            theta1 = angleArray[0]
            phi1 = angleArray[1]
            theta2 = angleArray[2]
            phi2 = angleArray[3]
            x = angleArray[4]
            y = angleArray[5]     
#        print "current angles", theta1, phi1, theta2, phi2

        penalty, phi1, phi2 = quadraticPenaltyForOutsideTanRange(phi1, phi2)

        warpPoint1 = getWarpPointFromAnglesVariableCenter(arrShape, theta1, phi1, x, y)
        warpPoint2 = getWarpPointFromAnglesVariableCenter(arrShape, theta2, phi2, x, y)

        warpedArray = warp2DArray(arr, warpPoint1, warpPoint2, numSteps=20)

        diffArray = batchAndDifferentiate(warpedArray, [(1, True), (1, True), (1, False)])

        totalArrayIntensity = getTotalArrayIntensity(diffArray)
#        print "intensity", totalArrayIntensity

        return totalArrayIntensity + penalty

    return evaluateAngle    


def getInitialAngleArray(rightAngleOnly=True):
    if rightAngleOnly:
        theta1 = random.random()*2*pi
        phi1 = random.random()*pi-pi/2

        theta2 = theta1 + pi/2
        phi2 = random.random()*pi-pi/2

    else:
        theta1 = random.random()*2*pi
        phi1 = random.random()*pi-pi/2

        theta2 = random.random()*2*pi
        phi2 = random.random()*pi-pi/2

    return np.array([theta1, phi1, theta2, phi2])

def findBestWarpPoints(arr):
    evaluateAngle = evaluateAngleMaker(arr)
    initialAngleArray = getInitialAngleArray(rightAngleOnly=True)

    xopt, nfeval, rc = fmin_tnc(evaluateAngle, initialAngleArray, approx_grad=True, epsilon=1e-2)

    return xopt

# second returnValue is rollover, third returnValue is an increment happening
def incrementList(l, maxVals):
    assert len(l) == len(maxVals)

    returnL = l[:]

    if returnL == []:
        return [], True

    else:

        returnL[0] += 1

        if returnL[0] == maxVals[0]:
            recursionResult = incrementList(returnL[1:], maxVals[1:])
            return [0] + recursionResult[0], recursionResult[1]

        else:
            return returnL, False

def product(l):
    returnVal = 1
    for i in l:
        returnVal *= i

    return returnVal

def bruteForceSearch(f, ranges, Ns):
    d = len(ranges)

    listOfLinspaces = [np.linspace(r[0], r[1], n) for r, n in zip(ranges, Ns)]

    currentIndices = [0]*d

    bestVal = float("Inf")
    bestLoc = None

    totalNum = product(Ns)
    counter = 0

    while True:
        if counter % 1000 == 0:
            print(counter, "/", totalNum)

        currentLoc = [x[i] for x, i in zip(listOfLinspaces, currentIndices)]

        currentVal = f(currentLoc)

        if currentVal < bestVal:
            bestVal = currentVal
            bestLoc = currentLoc

        currentIndices, done = incrementList(currentIndices, Ns)

        if done:
            break

        counter += 1

    return bestLoc, bestVal

def findBestWarpPointsBruteForce(arr):
    evaluateAngle = evaluateAngleMakerVariableCenter(arr)

    arrShape = arr.shape

    ranges = [(0, 2*pi), (-pi/2+1e-2, pi/2-1e-2), (-pi/2+1e-2, pi/2-1e-2), (0.33, 0.33), (0.5, 0.5)]
    Ns = [20, 10, 10, 1, 1]

    xopt, fopt = bruteForceSearch(evaluateAngle, ranges, Ns)

    return xopt

def average(x):
    return sum(x)/len(x)

def distanceBetween(x, y):
    return sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def colorStack(vid):
    colorStackedVid = np.swapaxes(np.reshape(np.swapaxes(vid, 1, 3), \
        (vid.shape[0]*3, vid.shape[2], vid.shape[1])), 1, 2)

    print(colorStackedVid.shape)

    return colorStackedVid

def pixelAngle(arr, pixel1, pixel2):
    history1 = arr[pixel1[0]][pixel1[1]]
    history2 = arr[pixel2[0]][pixel2[1]]

    flatHistory1 = history1.flatten()
    flatHistory2 = history2.flatten()

    return np.dot(flatHistory1, flatHistory2)/\
        (np.linalg.norm(flatHistory1)*np.linalg.norm(flatHistory2))

def extractUniquePixels(arr):
    swappedArr = np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)

    listOfGoodPixels = []

    for i in range(swappedArr.shape[0]):
        print(i, "/", swappedArr.shape[0])
        for j in range(swappedArr.shape[1]):
            pixel = swappedArr[i][j]

            pixelStdev = np.std(pixel, axis=0)

            if average(pixelStdev) > 30:
                listOfGoodPixels.append((i,j))

#    for pixel in listOfGoodPixels:
#        p.plot(pixel[0], pixel[1], "bo")
#    p.show()

    listOfUniquePixels = []

    print(len(listOfGoodPixels))

#    print(listOfGoodPixels)

    for i, pixel in enumerate(listOfGoodPixels):
        smallestDistance = 1e10
        nearestPixel = None

        if i % 1000 == 0:
            print(i, "/", len(listOfGoodPixels), len(listOfUniquePixels))

        unique = True

        for uniquePixel in listOfUniquePixels[-2000:]:
#            print(pixel, uniquePixel, distanceBetween(pixel, uniquePixel))
            if distanceBetween(pixel, uniquePixel) <= 8:            
                angle = pixelAngle(swappedArr, pixel, uniquePixel)
#                print(angle)
                if angle > 0.99:
                    unique = False

#        print(unique)

        if unique:
            listOfUniquePixels.append(pixel)


#        if nearestPixel == None:        
#            listOfUniquePixels.append(pixel)
#        else:
#            print(angle)

#            if angle < 0.9:
#                listOfUniquePixels.append(pixel)

    print(len(listOfUniquePixels))

#    listOfPixelHistories = []

#    for pixel in listOfUniquePixels:
#        listOfPixelHistories.append(swappedArr[pixel[0]][pixel[1]])

#    for pixel in listOfUniquePixels:
#        p.plot(pixel[0], pixel[1], "bo")
#    p.show()   

    return listOfUniquePixels

def extractUniquePixelsWithTree(arr):
    colorStackedArr = colorStack(arr)

    reshapedArr = np.reshape(colorStackedArr, (colorStackedArr.shape[0], 
        colorStackedArr.shape[1]*colorStackedArr.shape[2]))

    swappedReshapedArr = np.swapaxes(reshapedArr, 0, 1)

#    print(np.sum(colorStackedArr[0]))
#    print(np.sum(reshapedArr[0]))

#    assert sum(colorStackedArr[0]) == sum(reshapedArr[0])


    swappedArr = np.swapaxes(np.swapaxes(colorStackedArr, 0, 1), 1, 2)

    listOfGoodPixels = []

    for i in range(swappedArr.shape[0]):
        for j in range(swappedArr.shape[1]):
            pixel = swappedArr[i][j]

            pixelStdev = np.std(pixel, axis=0)

            if pixelStdev > 30:
                listOfGoodPixels.append((i, j))        

    tree = BallTree(swappedReshapedArr)

    TYPICAL_NEIGHBOR_COUNT = 5
    NUM_SAMPLES = 1000

    listOfDistances = []

    for _ in range(NUM_SAMPLES):
        randomIndex = random.choice(listOfGoodPixels)
        randomPixel = swappedArr[randomIndex[0]][randomIndex[1]]
        nClosestNeighbors = tree.query([randomPixel], k=TYPICAL_NEIGHBOR_COUNT+1, \
            sort_results=True, return_distance=True)
#        print(nClosestNeighbors)
#        print(nClosestNeighbors[0])
        nthClosestNeighborDistance = nClosestNeighbors[0][0][TYPICAL_NEIGHBOR_COUNT] 

        listOfDistances.append(nthClosestNeighborDistance)

#    print(listOfDistances, "listOfDistances")

    typicalDistance = np.median(listOfDistances)

    indicesToKeep = []

#    sys.exit()

    print(typicalDistance)


    for k, i in enumerate(listOfGoodPixels):
        if k % 100 == 0:
            print(k, "/", len(listOfGoodPixels), len(indicesToKeep))

        pixel = np.array([swappedArr[i[0]][i[1]]])
        neighbors = tree.query_radius(pixel, typicalDistance)[0]

        numNeighbors = len(neighbors)
#        print(neighbors)

#        if numNeighbors > 1:
#            print(numNeighbors)

#        print(numNeighbors)


        if numNeighbors >= 3:
            if numNeighbors <= TYPICAL_NEIGHBOR_COUNT*2:

                closeDistance = typicalDistance/5

                numCloseNeighbors = tree.query_radius(pixel, closeDistance, count_only=True) 

                probStay = 1/(1+numCloseNeighbors)

                if random.random() < probStay:
                    indicesToKeep.append(i)

    print(swappedArr.shape[0])
    print(len(indicesToKeep))

    goodPixelsOnlyArr = []

    for index in indicesToKeep:
        goodPixelsOnlyArr.append(swappedArr[index[0]][index[1]])

    return np.array(goodPixelsOnlyArr), indicesToKeep


if __name__ == "__main__":

    if batchMovie:
        pathToDir = "/Users/adamyedidia/walls/src/data_10_7_17/"
        path = pathToDir + "C0268.MP4"

        vid = imageio.get_reader(path,  'ffmpeg')

        VIDEO_TIME = "1:40"
        START_TIME = "0:00"
        END_TIME = "0:15"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        print(firstFrame, lastFrame)

        processVideo(vid, VIDEO_TIME, \
            np.array([(10, False), (10, False), (10, False), (1, False)]), \
            "doorway_vid", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame)

    if batchSmallerMovie:
        pathToDir = "/Users/adamyedidia/walls/src/"
        path = "smaller_movie.mov"

        vid = imageio.get_reader(path,  'ffmpeg')

        VIDEO_TIME = "0:30"
        START_TIME = "0:00"
        END_TIME = "0:30"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        print(firstFrame, lastFrame)

        processVideo(vid, VIDEO_TIME, \
            np.array([(1, True), (8, False), (8, False), (1, False)]), \
            "smaller_movie_batched_diff", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=True)        

    if batchRickMortyMovie:
        pathToDir = "/Users/adamyedidia/walls/src/"
        path = "steven.mp4"

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "0:26"
        START_TIME = "0:00"
        END_TIME = "0:26"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            np.array([(1, False), (30, False), (30, False), (1, False)]), \
            "steven_batched_coarse", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        


        print(firstFrame, lastFrame)

    if batchMulanMovie:
        pathToDir = "/Users/adamyedidia/walls/src/"
        path = "circle.3gp"

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "3:23"
        START_TIME = "0:00"
        END_TIME = "3:23"
        numFrames = len(vid)

        print(numFrames)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            np.array([(1, False), (3, False), (3, False), (1, False)]), \
            "circle_batched", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False, minY=8, maxY=40)        
        print(firstFrame, lastFrame)

    if batchOfficeMovie:
        pathToDir = "/Users/adamyedidia/walls/src/"
        path = "the_office.mp4"

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "0:31"
        START_TIME = "0:00"
        END_TIME = "0:30"
        numFrames = len(vid)

        print(numFrames)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            np.array([(3, False), (7, False), (7, False), (1, False)]), \
            "office_batched", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False, minY=8, maxY=40)        
        print(firstFrame, lastFrame)


    if processBlindDeconvVideo:
#        path = "/Users/adamyedidia/blind_deconv_videos/C0015.MP4"
        path = "/Users/adamyedidia/blind_deconv_videos/C0016.MP4"

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "0:33"
        START_TIME = "0:00"
        END_TIME = "0:24"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            np.array([(2, False), (15, False), (15, False), (1, False)]), \
            "hourglass", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        


        print(firstFrame, lastFrame)

    if processDualVideo:
        path = "/Users/adamyedidia/walls/src/IMG_0495.m4v"

#        path = "/Users/adamyedidia/blind_deconv_videos/C0015.MP4"
#        path = "/Users/adamyedidia/blind_deconv_videos/C0016.MP4"

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "2:47"
        START_TIME = "1:40"
        END_TIME = "1:50"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideo(vid, VIDEO_TIME, \
            np.array([(2, False), (15, False), (15, False), (1, False)]), \
            "blind_deconv_cardboard_1", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        


        print(firstFrame, lastFrame)

    if macarena:
        path = "/Users/adamyedidia/walls/src/IMG_0497.m4v"

#        path = "/Users/adamyedidia/blind_deconv_videos/C0015.MP4"
#        path = "/Users/adamyedidia/blind_deconv_videos/C0016.MP4"

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "4:19"
        START_TIME = "1:15"
        END_TIME = "1:30"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideo(vid, VIDEO_TIME, \
            np.array([(2, False), (15, False), (15, False), (1, False)]), \
            "macarena_dark_fixed", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        


        print(firstFrame, lastFrame)

    if test:
    #    FILE_NAME = "ir_video_rc_car.m4v"; VIDEO_TIME = "3:57"
        FILE_NAME = "hotel_vid.m4v"; VIDEO_TIME = "14:52"
        vid = imageio.get_reader(FILE_NAME,  'ffmpeg')

        START_TIME = "00:30"
        END_TIME = "00:40"
        numFrames = len(vid)
        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideo(vid, "14:52", [(10, True), (30, True), (30, True), (1, False)], "hotel_vid", \
            magnification=1000, firstFrame=firstFrame, lastFrame=lastFrame)

    if flag:
    #    imRaw = Image.open("japan_flag_garbled_new_1.png")
    #    imRaw = Image.open("texas_flag_garbled_1.png")
    #    imRaw = Image.open("texas_flag_garbled_dup_row.png")
    #    imRaw = Image.open("france_flag_garbled_1.png")
    #    imRaw = Image.open("us_flag_garbled_1.png")

        

        im = np.array(imRaw.convert("RGB")).astype(float)

        processedIm = batchAndDifferentiate(im, [(1, True), (1, True), (1, False)])



        print(processedIm)

        viewFrame(processedIm, 1e3, False)

    if downSample:

        dirName = "calibration"
        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/"
        im = pickle.load(open(path + "rectified2.p", "r"))

        processedIm = batchAndDifferentiate(im,[(10, False), (10, False), (1, False)])

    #    viewFrame(-processedIm, 1e3, False)

        pickle.dump(processedIm, open(path + "downsampled2.p", "w"))

    if downSample2:
        num = "268"
        path = "/Users/adamyedidia/walls/src/data_10_7_17/C0" + num + ".avi"

        vid = imageio.get_reader(path,  'ffmpeg')
        START_TIME = "00:00"
        END_TIME = "00:30"
        VIDEO_TIME = END_TIME
        numFrames = len(vid)
        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideo(vid, VIDEO_TIME, [(2, False), (10, False), (10, False), (1, False)], "downsampled_ceiling" + num, \
            magnification=3, firstFrame=firstFrame, lastFrame=lastFrame, toVideo=False)




    if raw:

        dirName = "texas_garbled"
        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified_very_wrong.p"
        im = pickle.load(open(path, "r"))

        processedIm = batchAndDifferentiate(im,[(100, True), (100, True), (1, False)])

        viewFrame(-processedIm, 1e3, False)

    if rawWithBlur:
        dirName = "uk_garbled"

        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified.p"
        im = pickle.load(open(path, "r"))

        processedIm = batchAndDifferentiate(im,[(10, True), (10, True), (1, False)])

    #    calibrationPath = "/Users/adamyedidia/flags_garbled/calibration/rectified.p"
    #
    #    calibrationIm = pickle.load(open(calibrationPath, "r"))

    #    calibrationProcessedIm = batchAndDifferentiate(calibrationIm, \
    #        [(10, True), (10, True), (1, False)])

    #    viewFrame(np.divide(processedIm, calibrationProcessedIm), 1e2, False)

        viewFrame(-processedIm, 2e3, False)

        blurredImage = blur2DImage(processedIm, 10)

        viewFrame(-blurredImage, 2e5, False)

    if viewDiff:
        path = "/Users/adamyedidia/flags/flag_of_france.png"

        imRaw = Image.open(path).convert("RGB")
        im = np.array(imRaw).astype(float)

        original = batchAndDifferentiate(im,[(45, False), (40, False), (1, False)])

        print(original.shape)

        viewFrame(original, 1e0, False)

    #    sys.exit()

    #    print "hi"

        dirName = "france_garbled"

        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified.p"
        im = pickle.load(open(path, "r"))

        processedIm = batchAndDifferentiate(im,[(100, True), (100, True), (1, False)])

        viewFrame(original + 1.5e3*processedIm, 1e0, True)

    #    viewFrame(-processedIm, 1e3, False)


    if downsampleWinnie:
#        path = "/Users/adamyedidia/walls/src/calvin_hobbes.jpg"
#        path = "/Users/adamyedidia/walls/src/winnie.png"
        path = "/Users/adamyedidia/walls/src/dora.png"
#        path = "/Users/adamyedidia/walls/src/shapes.png"

        imRaw = Image.open(path).convert("RGB")
        im = np.array(imRaw).astype(float)

#        processedIm = batchAndDifferentiate(im,[(5, False), (5, False), (1, False)])[:-5,:-3]
#        processedIm = batchAndDifferentiate(im,[(7, False), (7, False), (1, False)])[2:-7,4:-1]
#        processedIm = batchAndDifferentiate(im,[(15, False), (15, False), (1, False)])
#        processedIm = batchAndDifferentiate(im,[(19, False), (21, False), (1, False)])

        processedIm = batchAndDifferentiate(im,[(30, False), (30, False), (1, False)])

        print(im.shape)
        print(processedIm.shape)

        viewFrame(processedIm)

#        pickle.dump(processedIm, open("shapes_very_downsamples.p", "w"))
#        pickle.dump(processedIm, open("dora_very_downsampled.p", "w"))
        pickle.dump(processedIm, open("dora_extremely_downsampled.p", "w"))
#        pickle.dump(processedIm, open("dora_slightly_downsampled.p", "w"))
#        pickle.dump(processedIm, open("winnie_clipped.p", "w"))

    if rawWithSubtract:
        dirName = "texas_garbled"

        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified.p"
#        path = "/Users/adamyedidia/walls/100CANON/IMG_0065.jpg"

        im = pickle.load(open(path, "r"))

        processedIm = batchAndDifferentiate(im,[(10, True), (10, True), (1, False)])

        calibrationPath = "/Users/adamyedidia/flags_garbled/calibration/rectified2.p"

        calibrationIm = pickle.load(open(calibrationPath, "r"))

        calibrationProcessedIm = batchAndDifferentiate(calibrationIm, \
            [(10, True), (10, True), (1, False)])

        calibratedError = 255*3e-4*np.ones(calibrationProcessedIm.shape)+calibrationProcessedIm

#        viewFrame(-calibrationProcessedIm, 1e3)
    #    viewFrame(-processedIm, 1e3)
#        viewFrame(calibratedError, 3e3)

        calibrationSubtractedIm = -processedIm + calibratedError

    #    medfiltedIm = medfiltIm(calibrationSubtractedIm)
        medfiltedIm = calibrationSubtractedIm

        viewFrame(medfiltedIm,1e4)

    #    viewFrame(np.divide((-processedIm)**0.5, (-calibrationProcessedIm)**0.5), 1e2, False)

    #    viewFrame(-processedIm + 0.5*calibrationProcessedIm, 2e3, False)

    if weirdAngle:
    #    imRaw = Image.open("japan_flag_garbled_new_1.png")
    #    imRaw = Image.open("texas_flag_garbled_1.png")
    #    imRaw = Image.open("texas_flag_garbled_dup_row.png")
    #    imRaw = Image.open("france_flag_garbled_1.png")
    #    imRaw = Image.open("us_flag_garbled_1.png")

#        path = "/Users/adamyedidia/walls/100CANON/IMG_0098.jpg"

        path = "/Users/adamyedidia/walls/100CANON/IMG_0086.jpg"

#        path = "/Users/adamyedidia/Desktop/brit.png"
    
        imRaw = Image.open(path)

        fineIm = np.array(imRaw.convert("RGB")).astype(float)

        viewFrame(fineIm)

        print(fineIm.shape)

        im = batchAndDifferentiate(fineIm, [(12, False), (12, False), (1, False)])

#        warpPoint1 = np.array([150., 30000]) #2000, 650
#        warpPoint2 = np.array([-30000, 200])

#        warpPoint1 = np.array([0., 0]) #2000, 650
#        warpPoint2 = np.array([-200, 250])


 #       warpPoint1 = np.array([150., ])

#        warpPoint1 = np.array([150., 1800]) #2000, 650
#        warpPoint2 = np.array([-600, 400])

#        warpPoint1 = np.array([50, 800])
#        warpPoint2 = np.array([2000, 200])

#        warpPoint1 = np.array([500., 800])
#        warpPoint2 = np.array([600, -50])        

#        p.plot(warpPoint1[0], warpPoint1[1], "bo")
#        p.plot(warpPoint2[0], warpPoint2[1], "ro")

        viewFrame(im)

        print(im.shape)

        processedIm = batchAndDifferentiate(im,[(5, False), (5, False), (1, False)])
        
        lineIm = batchAndDifferentiate(batchAndDifferentiate(processedIm, \
            [(1, True), (1, True), (1, False)]), [(1, True), (1, True), (1, False)])

        viewFrame(lineIm, differenceImage=True, magnification=1e2)

        print(processedIm.shape)

#        viewFrame(processedIm)


#        print im.shape

#        viewFrame(im)

#        goodWarpedIm = warp2DArray(im, np.array([300, 10000]), np.array([-10000, 200]))
#        badWarpedIm = warp2DArray(im, np.array([300, -10000]), np.array([500, 200]))

#        processedGoodWarpedIm = batchAndDifferentiate(goodWarpedIm, [(5, True), (5, True), (1, False)])
 #       processedBadWarpedIm = batchAndDifferentiate(badWarpedIm, [(5, True), (5, True), (1, False)])

#        processedWarpedIm2 = batchAndDifferentiate(warpedIm, [(5, True), (5, True), (1, False)])

#        print "good", getTotalArrayIntensity(processedGoodWarpedIm)
 #       print "bad", getTotalArrayIntensity(processedBadWarpedIm)
  #      print "control", getTotalArrayIntensity(processedIm)

#        bestAngleArray = findBestWarpPointsBruteForce(processedIm)

#        theta1 = bestAngleArray[0]
#        phi1 = bestAngleArray[1]
 #       theta2 = theta1 + pi/2
 #       phi2 = bestAngleArray[2]
 #       x = bestAngleArray[3]
 #       y = bestAngleArray[4]

#        print "final angles", theta1, phi1, theta2, phi2, x, y

        arrShape = im.shape

#        warpPoint1 = getWarpPointFromAngles(arrShape, theta1, phi1)
 #       warpPoint2 = getWarpPointFromAngles(arrShape, theta2, phi2)        


        print("final warp points", warpPoint1, warpPoint2)

        warpedImage = warp2DArray(im, warpPoint1, warpPoint2, pixellationFactor=3)

        viewFrame(warpedImage)

        reconstructedImage = batchAndDifferentiate(warpedImage, [(2, True), (2, True), (1, False)])

        print(getTotalArrayIntensity(reconstructedImage))


#        pickle.dump((reconstructedImage, theta1, phi1, phi2), open("tilt_recovery_" + str(int(time.time())) + ".p", "w"))



        viewFrame(reconstructedImage, differenceImage=True, magnification=2e2)
        viewFrame(np.flip(-reconstructedImage,0), differenceImage=True, magnification=2e2)

#        viewFrame(np.swapaxes(-reconstructedImage,0,1), differenceImage=True, magnification=3e2)

#        viewFrame(warped)

#        viewFrame(processedIm, magnification=3e2, differenceImage=True)


    if weirdAngleSim:

#        imRaw = Image.open("/Users/adamyedidia/flags/flag_of_us.jpeg")
        imRaw = Image.open("/Users/adamyedidia/flags/flag_of_texas.png")

        scene = batchAndDifferentiate(np.array(imRaw.convert("RGB")).astype(float), [(10, False), (10, False), (1, False)])

#        viewFrame(scene)

#        obs = batchAndIntegrate(scene, [(1, True), (1, True), (1, False)])        

        sceneDimensions = scene.shape[:-1]

        print(sceneDimensions)

        occluderFrame = np.concatenate((np.concatenate((np.ones(sceneDimensions), np.zeros(sceneDimensions)), 0), \
            np.concatenate((np.ones(sceneDimensions), np.ones(sceneDimensions)), 0)), 1)

#        viewFrame(imageify(occluderFrame))

        obs = doFuncToEachChannel(lambda x: convolve2d(x, occluderFrame, mode="valid"), scene)

#        viewFrame(obs, 5e-4)
    
#        for i, logDistFromCenter in np.linspace(4, 0, 30):

        distFromCenter = 1000

        warpPoint1 = np.array([sceneDimensions[0]/2., sceneDimensions[1]/2+distFromCenter]) #2000, 650
        warpPoint2 = np.array([sceneDimensions[0]/2-distFromCenter, sceneDimensions[1]/2])

        rect = ptch.Rectangle((0,0), sceneDimensions[0], sceneDimensions[1], color="k")
        ax = p.gca()
        ax.add_patch(rect)

        p.plot(warpPoint1[0], warpPoint1[1], "ro")
        p.plot(warpPoint2[0], warpPoint2[1], "ro")
        p.show()

        viewFrame(obs, 9e-5, )

        preWarpedObs = preWarpArray(obs, warpPoint1, warpPoint2, numSamples=1)

        viewFrame(preWarpedObs, 9e-5)

        badRecoveredIm = batchAndDifferentiate(preWarpedObs, [(1, True), (1, True), (1, False)])

        viewFrame(badRecoveredIm, differenceImage=True, magnification=1)

#        warpedObs = warp2DArray(preWarpedObs, warpPoint1/20, warpPoint2/20, numSteps=25)
        warpedObs = warp2DArray(preWarpedObs, warpPoint1, warpPoint2, pixellationFactor=1, numSamples=3)

        viewFrame(warpedObs, differenceImage=True, magnification=9e-5)

        goodRecoveredIm = batchAndDifferentiate(warpedObs, [(1, True), (1, True), (1, False)])

        viewFrame(goodRecoveredIm, differenceImage=True, magnification=1)

    if weirdAngleSimMovie:

        imRaw = Image.open("/Users/adamyedidia/flags/flag_of_us.jpeg")
#        imRaw = Image.open("/Users/adamyedidia/flags/flag_of_texas.png")

        scene = batchAndDifferentiate(np.array(imRaw.convert("RGB")).astype(float), [(10, False), (10, False), (1, False)])

#        viewFrame(scene)

#        obs = batchAndIntegrate(scene, [(1, True), (1, True), (1, False)])        

        sceneDimensions = scene.shape[:-1]

        print(sceneDimensions)

        occluderFrame = np.concatenate((np.concatenate((np.ones(sceneDimensions), np.zeros(sceneDimensions)), 0), \
            np.concatenate((np.ones(sceneDimensions), np.ones(sceneDimensions)), 0)), 1)

#        viewFrame(imageify(occluderFrame))

        obs = doFuncToEachChannel(lambda x: convolve2d(x, occluderFrame, mode="valid"), scene)

#        viewFrame(obs, 5e-4)
    
        for i, logDistFromCenter in enumerate(np.linspace(4, 1.5, 100)):

            distFromCenter = 10**logDistFromCenter

            print(distFromCenter)

            warpPoint1 = np.array([sceneDimensions[0]/2., sceneDimensions[1]/2+distFromCenter]) #2000, 650
            warpPoint2 = np.array([sceneDimensions[0]/2-distFromCenter, sceneDimensions[1]/2])

            p.clf()
            rect = ptch.Rectangle((0,0), sceneDimensions[0], sceneDimensions[1], color="k")
            ax = p.gca()
            ax.add_patch(rect)

            p.plot(warpPoint1[0], warpPoint1[1], "ro")
            p.plot(warpPoint2[0], warpPoint2[1], "ro")
            p.savefig("warp_points_" + str(padIntegerWithZeros(i, 2)) + ".png")

#            viewFrame(obs, 9e-5)

            preWarpedObs = preWarpArray(obs, warpPoint1, warpPoint2, numSamples=1)

            viewFrame(preWarpedObs, magnification=9e-5, \
                filename="prewarp_" + str(padIntegerWithZeros(i, 2)) + ".png")

            badRecoveredIm = batchAndDifferentiate(preWarpedObs, [(1, True), (1, True), (1, False)])

            viewFrame(badRecoveredIm, differenceImage=True, magnification=1, 
                filename="bad_recovery_" + str(padIntegerWithZeros(i, 2)) + ".png")

    #        warpedObs = warp2DArray(preWarpedObs, warpPoint1/20, warpPoint2/20, numSteps=25)
            warpedObs = warp2DArray(preWarpedObs, warpPoint1, warpPoint2, pixellationFactor=1, numSamples=3)

            viewFrame(warpedObs, differenceImage=True, magnification=9e-5,
                filename="warp_obs_" + str(padIntegerWithZeros(i, 2)) + ".png")

            goodRecoveredIm = batchAndDifferentiate(warpedObs, [(1, True), (1, True), (1, False)])

            viewFrame(goodRecoveredIm, differenceImage=True, magnification=1,
                filename="good_recovery_" + str(padIntegerWithZeros(i, 2)) + ".png")

    if weirdAngleSimRecovery:
#        imRaw = Image.open("/Users/adamyedidia/flags/flag_of_texas.png")
        imRaw = Image.open("/Users/adamyedidia/flags/flag_of_us.jpeg")
#        imRaw = Image.open("/Users/adamyedidia/flags/flag_of_japan.png")


        scene = batchAndDifferentiate(np.array(imRaw.convert("RGB")).astype(float), [(10, False), (10, False), (1, False)])
 
        viewFrame(scene, differenceImage=True)
        sceneDimensions = scene.shape[:-1]

        print(sceneDimensions)

        occluderFrame = np.concatenate((np.concatenate((np.ones(sceneDimensions), np.zeros(sceneDimensions)), 0), \
            np.concatenate((np.ones(sceneDimensions), np.ones(sceneDimensions)), 0)), 1)

#        viewFrame(imageify(occluderFrame))

        obs = doFuncToEachChannel(lambda x: convolve2d(x, occluderFrame, mode="valid"), scene)

        warpPoint1 = np.array([-100, 200])
        warpPoint2 = np.array([200, 300])

        preWarpedObs = addNoise(preWarpArray(obs, warpPoint1, warpPoint2, numSamples=1), 0)

        viewFrame(preWarpedObs, magnification=9e-5, filename="pass")     
        p.plot(warpPoint1[1], warpPoint1[0], "ko")
        p.plot(warpPoint2[1], warpPoint2[0], "ko")
        p.show()       

        badRecoveredIm = batchAndDifferentiate(preWarpedObs, [(1, True), (1, True), (1, False)])
        lines = batchAndDifferentiate(badRecoveredIm, [(1, True), (1, True), (1, False)])

        viewFrame(badRecoveredIm, differenceImage=False, magnification=1)

        viewFrame(badRecoveredIm, differenceImage=False, magnification=1, filename="pass")
        p.plot(warpPoint1[1], warpPoint1[0], "ko")
        p.plot(warpPoint2[1], warpPoint2[0], "ko")
        p.show()        

        viewFrame(lines, differenceImage=False, magnification=1)

        viewFrame(lines, differenceImage=False, magnification=1, filename="pass")
        p.plot(warpPoint1[1], warpPoint1[0], "ko")
        p.plot(warpPoint2[1], warpPoint2[0], "ko")
        p.show()      

        warpedObs = warp2DArray(preWarpedObs, warpPoint1, warpPoint2, pixellationFactor=1, numSamples=3)

        viewFrame(warpedObs, differenceImage=True, magnification=3e-5)

        goodRecoveredIm = batchAndDifferentiate(warpedObs, [(1, True), (1, True), (1, False)])

        viewFrame(goodRecoveredIm, differenceImage=True, magnification=1)

    if video36225:
        path = "/Users/adamyedidia/walls/src/IMG_0503.m4v"

#        path = "/Users/adamyedidia/blind_deconv_videos/C0015.MP4"
#        path = "/Users/adamyedidia/blind_deconv_videos/C0016.MP4"

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "1:42"
        START_TIME = "0:34"
        END_TIME = "1:00"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideo(vid, VIDEO_TIME, \
            np.array([(2, False), (15, False), (15, False), (1, False)]), \
            "36225_bright_fixed", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        


        print(firstFrame, lastFrame)

    if orange:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9422.MOV"

        vid = imageio.get_reader(path, 'ffmpeg')

#        viewFrame(np.array(vid.get_data(500)).astype(float))

        VIDEO_TIME = "1:00"
        START_TIME = "0:30"
        END_TIME = "1:00"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (15, False), (15, False), (1, False)], \
            "36225_bright_fixed", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

    if orange:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9422.MOV"

        vid = imageio.get_reader(path, 'ffmpeg')

#        viewFrame(np.array(vid.get_data(500)).astype(float))

        VIDEO_TIME = "1:00"
        START_TIME = "0:30"
        END_TIME = "1:00"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (15, False), (15, False), (1, False)], \
            "36225_bright_fixed", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

        print(firstFrame, lastFrame)

    if bld66:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9424.MOV"

        vid = imageio.get_reader(path, 'ffmpeg')

#        viewFrame(np.array(vid.get_data(500)).astype(float))

        VIDEO_TIME = "1:51"
        START_TIME = "1:21"
        END_TIME = "1:51"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (15, False), (15, False), (1, False)], \
            "bld66", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

        print(firstFrame, lastFrame)        

    if bld34:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9418.MOV"

        vid = imageio.get_reader(path, 'ffmpeg')

#        viewFrame(np.array(vid.get_data(500)).astype(float))

        VIDEO_TIME = "3:29"
        START_TIME = "0:13"
        END_TIME = "0:43"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (15, False), (15, False), (1, False)], \
            "stata", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

        print(firstFrame, lastFrame)           

    if stata:
        path = "/Users/adamyedidia/walls/src/IMG_0541.m4v"    

        vid = imageio.get_reader(path, 'ffmpeg')

        VIDEO_TIME = "3:29"
        START_TIME = "0:13"
        END_TIME = "0:43"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (10, False), (10, False), (1, False)], \
            "stata", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

        print(firstFrame, lastFrame)           

    if fan:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9435.MOV"    

        vid = imageio.get_reader(path, 'ffmpeg')

        VIDEO_TIME = "2:33"
        START_TIME = "1:20"
        END_TIME = "1:53"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        print(firstFrame, lastFrame)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (10, False), (10, False), (1, False)], \
            "fan", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

        print(firstFrame, lastFrame)           

    if fan_fine:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9435.MOV"    

        vid = imageio.get_reader(path, 'ffmpeg')

        VIDEO_TIME = "2:33"
        START_TIME = "1:20"
        END_TIME = "1:53"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        print(firstFrame, lastFrame)

        processVideoCheap(vid, VIDEO_TIME, \
            [(1, False), (10, False), (10, False), (1, False)], \
            "fan_fine", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

        print(firstFrame, lastFrame)          

    if fan_monitor:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9432.MOV"

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "0:35"
        START_TIME = "0:03"        
        END_TIME = "0:29"

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(1, False), (10, False), (10, False), (1, False)], \
            "fan_monitor_fine", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

    if plant:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9437.MOV"    

        vid = imageio.get_reader(path, 'ffmpeg')

        VIDEO_TIME = "1:16"
        START_TIME = "0:36"
        END_TIME = "1:16"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        print(firstFrame, lastFrame)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (10, False), (10, False), (1, False)], \
            "plant", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

        print(firstFrame, lastFrame)

    if plant_fine:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9437.MOV"    

        vid = imageio.get_reader(path, 'ffmpeg')

        VIDEO_TIME = "1:16"
        START_TIME = "0:36"
        END_TIME = "1:16"
        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        print(firstFrame, lastFrame)

        processVideoCheap(vid, VIDEO_TIME, \
            [(1, False), (10, False), (10, False), (1, False)], \
            "plant_fine", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)        

        print(firstFrame, lastFrame)


    if plant_monitor:
        path = "/Users/adamyedidia/walls/src/movies/MVI_9438.MOV"

        vid = imageio.get_reader(path, 'ffmpeg')

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "0:33"
        START_TIME = "0:03"        
        END_TIME = "0:29"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (10, False), (10, False), (1, False)], \
            "plant_monitor", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)      

    if glass_rose:
        path = "/Users/adamyedidia/walls/src/glass_rose.mov"

        vid = imageio.get_reader(path, 'ffmpeg')

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "0:28"
        START_TIME = "0:00"        
        END_TIME = "0:26"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (8, False), (8, False), (1, False)], \
            "glass_rose", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False, minY=50, maxY=90, minX=47, maxX=138)      

#        processVideoCheap(vid, VIDEO_TIME, \
#            [(2, False), (2, False), (2, False), (1, False)], \
#            "glass_rose", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
#            toVideo=False, minY=200, maxY=360, minX=190, maxX=550)      

    if glass_rose_2:
        path = "/Users/adamyedidia/walls/src/glass_rose_calibration.m4v"

        vid = imageio.get_reader(path, 'ffmpeg')

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "0:24"
        START_TIME = "0:03"        
        END_TIME = "0:14"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(1, False), (8, False), (8, False), (1, False)], \
            "glass_rose_calibration", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False, minY=45, maxY=100, minX=0, maxX=85)              

    if impulse_movie:
        path = "/Users/adamyedidia/walls/src/impulse_movie.mp4"

        vid = imageio.get_reader(path, 'ffmpeg')

        vid = imageio.get_reader(path, 'ffmpeg')
        VIDEO_TIME = "0:11"
        START_TIME = "0:00"        
        END_TIME = "0:11"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(2, False), (80, False), (80, False), (1, False)], \
            "impulse_movie", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)   

    if matthew_wall:
        path = "/Users/adamyedidia/walls/src/IMG_0571.m4v"

        vid = imageio.get_reader(path, 'ffmpeg')

        frame1 = vid.get_data(656)

        viewFrame(frame1)

        frame2 = vid.get_data(924)

        viewFrame(frame2)

        viewFrame(frame2 - frame1, differenceImage=True, adaptiveScaling=True)

        diffFrame = frame2 - frame1

        diffFrameBatched = batchAndDifferentiate(diffFrame, [(50, False), (50, False), (1, False)])

        viewFrame(diffFrameBatched, differenceImage=True, adaptiveScaling=True, magnification=0.1)

    if darpa_vid:
        path = "/Users/adamyedidia/walls/src/fusion_exp/MVI_9459.MOV"

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "0:59"
        START_TIME = "0:08"        
        END_TIME = "0:45"

        numFrames = len(vid)

        print(numFrames)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(5, False), (10, False), (10, False), (1, False)], \
            "darpa_vid_2", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)                   

    if darpa_gt:
        path = "/Users/adamyedidia/walls/src/ground_truth.m4v"

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "0:50"
        START_TIME = "0:10"        
        END_TIME = "0:50"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(5, False), (10, False), (10, False), (1, False)], \
            "darpa_vid_gt", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)                   

    if obama:
        path = "/Users/adamyedidia/walls/src/obama.mp4"

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "9:00"
        START_TIME = "0:00"        
        END_TIME = "7:00"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(10, False), (5, False), (5, False), (1, False)], \
            "obama_long", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)                   

    if darpa_fan:
        path = "/Users/adamyedidia/DARPA_demo/C0283.MP4"

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "2:05"
        START_TIME = "1:28"        
        END_TIME = "2:05"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(5, False), (10, False), (10, False), (1, False)], \
            "fan_darpa_longer", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)     

    if darpa_iphone_fan:
        path = "/Users/adamyedidia/DARPA_demo/C0278.MP4"

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "0:38"
        START_TIME = "0:00"        
        END_TIME = "0:38"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(5, False), (10, False), (10, False), (1, False)], \
            "darpa_iphone_fan", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)     


    if darpa_mannequin:
        path = "/Users/adamyedidia/DARPA_demo/C0281.MP4"

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "0:38"
        START_TIME = "0:00"        
        END_TIME = "0:38"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(5, False), (10, False), (10, False), (1, False)], \
            "mannequin_darpa", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)     

    if particle_vid:
        path = "/Users/adamyedidia/walls/src/IMG_0864.m4v"

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "0:29"
        START_TIME = "0:00"        
        END_TIME = "0:06"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(1, False), (3, False), (3, False), (1, False)], \
            "particle_vid", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)     

    if sidewalk:
        path = "/Users/adamyedidia/walls/src/python3_files/sidewalk.m4v"

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "0:38"
        START_TIME = "0:15"        
        END_TIME = "0:25"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheap(vid, VIDEO_TIME, \
            [(1, False), (3, False), (3, False), (1, False)], \
            "sidewalk_vid", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False)             

    if lion_king_bottle:
        path = "/Users/adamyedidia/walls/src/python3_files/lion_king.m4v"

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "4:08"
        START_TIME = "0:05"        
        END_TIME = "4:00"

        numFrames = len(vid)

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

#        print(firstFrame.shape)

        processVideoCheap(vid, VIDEO_TIME, \
            [(3, False), (10, False), (10, False), (1, False)], \
            "lion_king", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False, minX=20, maxX=87, minY=42, maxY=191)             

    if lion_king_bottle_sparsified:
        generateSample = True
        if generateSample:
            path = "/Users/adamyedidia/walls/src/python3_files/lion_king.m4v"

            vid = imageio.get_reader(path, 'ffmpeg')
            
            VIDEO_TIME = "4:08"
            START_TIME = "0:05"        
            END_TIME = "4:00"

            sampleStart = "0:15"
            sampleEnd = "0:30"

            numFrames = len(vid)

            firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
            lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

            firstFrameSample = getFrameAtTime(sampleStart, VIDEO_TIME, numFrames)
            lastFrameSample = getFrameAtTime(sampleEnd, VIDEO_TIME, numFrames)

            multiple = 1

    #        print(firstFrame.shape)

            arr = processVideoCheap(vid, VIDEO_TIME, \
                [(3, False), (multiple, False), (multiple, False), (1, False)], \
                "lion_king", magnification=1, firstFrame=firstFrameSample, lastFrame=lastFrameSample, 
                toVideo=False, minX=int(200/multiple), maxX=int(870/multiple), minY=int(420/multiple), maxY=int(1910/multiple), returnSomething=True)             
        else:
            arr = pickle.load(open("lion_king.p", "rb"))


        pixelMovie, pixelLocs = extractUniquePixelsWithTree(arr)

        pickle.dump(pixelLocs, open("unique_pixels_lion_king_new.p", "wb"))
        pickle.dump(pixelMovie, open("unique_pixels_lion_king_movie.p", "wb"))

        for i in pixelLocs:
            p.plot(i[1], i[0], "bo")
        viewFrame(arr[100], relax=True)


    if lion_king_bottle_build_array:
        path = "/Users/adamyedidia/walls/src/python3_files/lion_king.m4v"
        vid = imageio.get_reader(path, 'ffmpeg')

        listOfPixels = random.sample(pickle.load(open("unique_pixels_lion_king_new.p", "rb"))[:60000], 10000)
        numFrames = len(vid)

        pickle.dump(listOfPixels, open("selected_pixels_lion_king_10k.p", "wb"))

#        for i,pixel in enumerate(listOfPixels[:3000]):
#            print(i)
#            p.plot(pixel[0], pixel[1], "bo")
#        p.show()

        vid = imageio.get_reader(path, 'ffmpeg')
        
        VIDEO_TIME = "4:08"
        START_TIME = "0:05"        
        END_TIME = "4:00"

        firstFrame = getFrameAtTime(START_TIME, VIDEO_TIME, numFrames)
        lastFrame = getFrameAtTime(END_TIME, VIDEO_TIME, numFrames)

        processVideoCheapWithPixels(vid, listOfPixels, VIDEO_TIME, \
            [(10, False), (1, False), (1, False), (1, False)], \
            "lion_king_10k", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame, 
            toVideo=False, minX=200, maxX=870, minY=420, maxY=1910, returnSomething=False)             


    if colorStackTest:
        vid = pickle.load(open("particle_vid.p", "rb"))

        print(vid.shape)

        viewFrame(vid[100])

        colorStackedVid = colorStack(vid)

        print(colorStackedVid.shape)

        viewFrame(imageify(colorStackedVid[300]), adaptiveScaling=True)


