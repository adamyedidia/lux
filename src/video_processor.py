from __future__ import division
import pylab
import imageio
import numpy as np
from math import acos, pi, cos, sin, log, floor, ceil
import matplotlib.pyplot as p
import matplotlib.cm as cm
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

test = False
flag = False
raw = False
viewDiff = False
downSample = False
downSample2 = False
rawWithSubtract = False
rawWithBlur = False
batchMovie = False
downsampleWinnie = False

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
    colonIndex = string.find(timeString, ":")
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

#    viewFrame(arr, 1, False)

    # take gradients
#    print "Differentiating..."
    for i in range(dim - 1, -1, -1):
        if listOfResponses[i][1]:
            arr = np.gradient(arr, axis=i)

#            viewFrame(arr, 1e1, True)

#    arr = blur2DImage(arr, 5)

    return arr

def convertArrayToVideo(arr, magnification, filename, frameRate):
    assert len(arr.shape) == 4

    print arr.shape
    numFrames = arr.shape[0]
    logNumFrames = int(floor(log(numFrames, 10)))+1
    print "logNumFrames", logNumFrames

    for i, frame in enumerate(arr):
        print frame.shape
        print type(frame[0][0][0])
        viewFrame(frame, magnification=magnification, filename="video_trash/" + filename + "_" + \
            padIntegerWithZeros(i, logNumFrames) + ".png", differenceImage=True)

    numDigits = ceil(log(len(arr), 10))

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

    numFramesInOriginalVideo = len(vid)
    originalFrameRate = numFramesInOriginalVideo / convertTimeToSeconds(vidLength)

    newFrameRate = originalFrameRate / listOfResponses[0][0]

    if toVideo:
        convertArrayToVideo(arr, magnification, filename, newFrameRate)
    else:
        pickle.dump(arr, open(filename + ".p", "w"))

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

        print firstFrame, lastFrame

        processVideo(vid, VIDEO_TIME, \
            np.array([(10, False), (10, False), (10, False), (1, False)]), \
            "doorway_vid", magnification=1, firstFrame=firstFrame, lastFrame=lastFrame)

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
        imRaw = Image.open("japan_flag_garbled_new_1.png")
    #    imRaw = Image.open("texas_flag_garbled_1.png")
    #    imRaw = Image.open("texas_flag_garbled_dup_row.png")
    #    imRaw = Image.open("france_flag_garbled_1.png")
    #    imRaw = Image.open("us_flag_garbled_1.png")


        im = np.array(imRaw.convert("RGB")).astype(float)

        processedIm = batchAndDifferentiate(im, [(20, True), (20, True), (1, False)])



        print processedIm

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

        print original.shape

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
        processedIm = batchAndDifferentiate(im,[(15, False), (15, False), (1, False)])
#        processedIm = batchAndDifferentiate(im,[(19, False), (21, False), (1, False)])

        print im.shape
        print processedIm.shape

        viewFrame(processedIm)

#        pickle.dump(processedIm, open("shapes_very_downsamples.p", "w"))
        pickle.dump(processedIm, open("dora_very_downsampled.p", "w"))
#        pickle.dump(processedIm, open("dora_slightly_downsampled.p", "w"))
#        pickle.dump(processedIm, open("winnie_clipped.p", "w"))

    if rawWithSubtract:
        dirName = "texas_garbled"

        path = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified.p"
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
