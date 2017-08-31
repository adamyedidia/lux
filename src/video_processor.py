from __future__ import division
import pylab
import imageio
import numpy as np
from math import acos, pi, cos, sin, log, floor
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

test = False
flag = False
raw = True
downSample = False
rawWithSubtract = False

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

    print len(listOfBigFrames)
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
    print "Batching..."
    for i in range(dim):
        arr = batchArrayAlongAxis(arr, i, listOfResponses[i][0])

#    viewFrame(arr, 1, False)

    # take gradients
    print "Differentiating..."
    for i in range(dim - 1, -1, -1):
        if listOfResponses[i][1]:
            arr = np.gradient(arr, axis=i)

#            viewFrame(arr, 1e1, True)


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

    os.system("ffmpeg -r " + str(frameRate) + " -f image2 -s 500x500 " + \
        "-i video_trash/" + filename + "_%02d.png " + \
        "-vcodec libx264 -crf 25 -pix_fmt yuv420p " + filename + ".mp4")
    os.system("y")

def getFrameAtTime(frameTime, videoTime, numFrames):
    return int(convertTimeToSeconds(frameTime) / \
        convertTimeToSeconds(videoTime) * numFrames)

def processVideo(vid, vidLength, listOfResponses, filename, magnification=1, \
    firstFrame=0, lastFrame=None):

    arr = turnVideoIntoArray(vid, firstFrame, lastFrame)
    arr = batchAndDifferentiate(arr, listOfResponses)

    numFramesInOriginalVideo = len(vid)
    originalFrameRate = numFramesInOriginalVideo / convertTimeToSeconds(vidLength)

    newFrameRate = originalFrameRate / listOfResponses[0][0]

    convertArrayToVideo(arr, magnification, filename, newFrameRate)

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

    processedIm = batchAndDifferentiate(im, [(20, True), (20, True), (1, False)])



    print processedIm

    viewFrame(processedIm, 1e2, False)

if downSample:

    dirName = "temesvar_garbled"
    path = "/Users/adamyedidia/flags_garbled/" + dirName + "/"
    im = pickle.load(open(path + "rectified.p", "r"))

    processedIm = batchAndDifferentiate(im,[(10, False), (10, False), (1, False)])

#    viewFrame(-processedIm, 1e3, False)

    pickle.dump(processedIm, open(path + "downsampled.p", "w"))

if raw:

    dirName = "texas_garbled"
    path = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified_very_wrong.p"
    im = pickle.load(open(path, "r"))

    processedIm = batchAndDifferentiate(im,[(100, True), (100, True), (1, False)])

    viewFrame(-processedIm, 1e3, False)

if rawWithSubtract:
    dirName = "uk_garbled"

    path = "/Users/adamyedidia/flags_garbled/" + dirName + "/rectified.p"
    im = pickle.load(open(path, "r"))

    processedIm = batchAndDifferentiate(im,[(100, True), (100, True), (1, False)])

    calibrationPath = "/Users/adamyedidia/flags_garbled/calibration/rectified.p"

    calibrationIm = pickle.load(open(calibrationPath, "r"))

    calibrationProcessedIm = batchAndDifferentiate(calibrationIm, \
        [(100, True), (100, True), (1, False)])

#    viewFrame(np.divide(processedIm, calibrationProcessedIm), 1e2, False)

    viewFrame(-processedIm + 0.5*calibrationProcessedIm, 2e3, False)
