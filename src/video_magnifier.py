from __future__ import division
import pylab
import imageio
import numpy as np
from math import acos, pi, cos, sin
import matplotlib.pyplot as p
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pickle
from PIL import Image
from PIL import ImageFilter
import sys

#filename = 'pokemon_video_2.m4v'
#vid = imageio.get_reader(filename,  'ffmpeg')

#numFrames = len(vid)
#print numFrames

makeBatches = False
exportVideoAsListOfFrames = False

def turnVideoIntoListOfFrames(vid, firstFrame=0, lastFrame=None):
    listOfFrames = []
    numFrames = len(vid)

    if lastFrame == None:
        lastFrame = numFrames

    for i in range(firstFrame, lastFrame): #range(int(numFrames)): #i in range(400, 500):
        print i-firstFrame, "/", lastFrame - firstFrame
        try:
            im = vid.get_data(i)
        except:
            im = vid[i]
        frame = np.array(im).astype(float)
        listOfFrames.append(frame)

    return listOfFrames

def average(l):
#    print "sum", sum(l)
    return sum(l)/len(l)

def batchIntoBigFrames(listOfFrames, batchSize):
    listOfBigFrames = []
    listOfBigFramesSquared = []

    numFrames = len(listOfFrames)

    zeroFrame = np.zeros(listOfFrames[0].shape)

    firstEntries = []
    firstSquaredEntries = []

    for i in range(numFrames):
        print i, "/", numFrames, "batching"
        if i % batchSize == 0:
            if len(listOfBigFrames) > 0:
                #print "compare", listOfBigFrames[-1], np.square(listOfBigFrames[-1])
                listOfBigFramesSquared.append(np.square(listOfBigFrames[-1]))
                firstSquaredEntries.append(listOfBigFramesSquared[-1][0][0][0])
                firstEntries.append(listOfBigFrames[-1][0][0][0])


            listOfBigFrames.append(zeroFrame.copy())


        listOfBigFrames[-1] += listOfFrames[i]

    listOfBigFramesSquared.append(np.square(listOfBigFrames[-1]))
    firstEntries.append(listOfBigFrames[-1][0][0][0])
    firstSquaredEntries.append(listOfBigFramesSquared[-1][0][0][0])

    print firstEntries
    print firstSquaredEntries

    return listOfBigFrames, listOfBigFramesSquared

def viewFrame(frame, magnification=1, differenceImage=False, meanSubtraction=False, \
    absoluteMeanSubtraction=False, filename=None):
    frameShape = frame.shape

    if meanSubtraction:
        numPixels = frame.shape[0]*frame.shape[1]
        averagePixel = np.sum(np.sum(frame, 0), 0)/numPixels

        adjustedFrame = np.zeros(frameShape)

        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if absoluteMeanSubtraction:
                    adjustedFrame[i][j] = abs(frame[i][j] - averagePixel)
                else:
                    adjustedFrame[i][j] = frame[i][j] - averagePixel

#        frame -= arrayOfAveragePixel
    else:
        adjustedFrame = frame.copy()

#    print frame, magnification
#    print type(frame[0][0][0]), type(magnification)
    adjustedFrame = adjustedFrame*magnification

#    print adjustedFrame

    if differenceImage:
        adjustedFrame += np.full(shape=frameShape, \
            fill_value=128.)

    coercedFrame = np.minimum(np.maximum(adjustedFrame, np.zeros(frameShape)), \
        np.full(shape=frameShape, fill_value=255))

    #print coercedFrame.astype(np.uint8)

#    print coercedFrame
#    print coercedFrame.astype(np.uint8)

    pylab.imshow(coercedFrame.astype(np.uint8))



    if filename == None:
        p.show()
    else:
        p.savefig(filename)

def viewFrameR(frameR, magnification=1, differenceImage=False, fileName=None):
    frameShape = frameR.shape
    rearrangedFrame = np.array([frameR, np.zeros(shape=frameShape), np.zeros(shape=frameShape)])

    frame = np.swapaxes(np.swapaxes(rearrangedFrame, 1, 2), 0, 2)

    viewFrame(frame, magnification, differenceImage, fileName)

def viewFlatFrame(flattenedFrame, height, magnification=1, \
    differenceImage=False, filename=None):

    frame = np.array([flattenedFrame]*height)
#    print frame.shape

    viewFrame(frame, differenceImage=differenceImage, magnification=magnification,
        filename=filename)

def playFrameByFrame(listOfFrames):

    numFrames = len(listOfFrames)

    for i in range(int(numFrames)):
#        print i
        frame = listOfFrames[i]
    #    print np.array(im)
        viewFrame(frame)

if makeBatches:

    BATCH_SIZE = 10

    listOfFrames = turnVideoIntoListOfFrames(vid)
    print "list of frames created"
    listOfBigFrames, listOfBigFramesSquared = batchIntoBigFrames(listOfFrames, BATCH_SIZE)

    means = average(listOfBigFrames)
#    print listOfBigFramesSquared
#    print listOfBigFrames

    print "E[x^2]", average(listOfBigFramesSquared)
    print "E[x]^2", np.square(average(listOfBigFrames))

    stdevs = np.sqrt(average(listOfBigFramesSquared) - \
        np.square(average(listOfBigFrames)))

    pickle.dump((listOfBigFrames, means, stdevs), open("batched_pokemon_video_" + \
        sys.argv[1] + "_" + sys.argv[2] + ".p", "w"))

    print len(listOfBigFrames)
    print "frames batched"
    #playFrameByFrame(listOfBigFrames)

if exportVideoAsListOfFrames:
    FILE_NAME = "smaller_movie.mov"
    vid = imageio.get_reader(FILE_NAME,  'ffmpeg')

    listOfFrames = turnVideoIntoListOfFrames(vid)
    pickle.dump(listOfFrames, open("short_video_pickle.p", "w"))
