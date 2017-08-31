from __future__ import division
from PIL import Image
import numpy as np
import random
from math import cos, pi, sin

def generateRandomOccluder(length, pSwitch):
    currentValue = 1

    occluderList = []
    for i in range(length):
        if random.random() < pSwitch:
            currentValue = 1 - currentValue

        occluderList.append(currentValue)

    return np.array(occluderList)


#listOfPixels = [(0,0,0),(255,255),(0,0,0),(255,255),(0,0,0),(255,255), \
#        (0,0,0),(255,255),(0,0,0)]

listOfPixels = []

COLUMN_WIDTH = 100

WIDTH = 1000
HEIGHT = 700

BACKGROUND = generateRandomOccluder(WIDTH, 0.005)

def padNumber(x):
    if x < 10:
        return "00" + str(x)
    elif x < 100:
        return "0" + str(x)
    return str(x)

def makeFrame(frameNum, columnLoc):
    horizontalRow = []

    for i in range(WIDTH):
        if i >= columnLoc and i < columnLoc + COLUMN_WIDTH:
            horizontalRow.append(128)
        else:
            horizontalRow.append(255*BACKGROUND[i])

    listOfPixels = horizontalRow * HEIGHT
    im = Image.new('L', (WIDTH, HEIGHT))
    im.putdata(listOfPixels)
    im.save("artificial_video/frame_" + padNumber(frameNum) + ".png")

columnLoc = 450
columnSpeed = 0


MOVIE_LENGTH = 1000


for frameNum in range(MOVIE_LENGTH):
    print frameNum
    columnLoc = 450 * sin(frameNum/1000*2*pi) + 450
#    print columnLoc

    makeFrame(frameNum, columnLoc)
