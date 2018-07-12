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
from video_processor import actOnRGB, batchAndDifferentiate, padIntegerWithZeros
import os
import string
import pickle
from scipy.signal import convolve2d, medfilt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate as rotateImage
import sys

def integrateArray(arr):
    arrShape = arr.shape

    return np.dot(np.dot(np.triu(np.ones(arrShape[0])), arr), \
        np.transpose(np.triu(np.ones(arrShape[1]))))

#def padArrayWithZeros()

#def rotateImage(arr):

#    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
#    dst = cv2.warpAffine(arr,M,(cols,rows))


def getNegative(val):
    if val < 0:
        return -val

    else:
        return 0


def getNegativeSquared(val):
    if val < 0:
        return val*val

    else:
        return 0

flag = False
flagMovie = True

if __name__ == "__main__":
    if flag:
    #    imRaw = Image.open("japan_flag_garbled_new_1.png")
    #    imRaw = Image.open("texas_flag_garbled_1.png")
    #    imRaw = Image.open("texas_flag_garbled_dup_row.png")
    #    imRaw = Image.open("france_flag_garbled_1.png")
        path = "/Users/adamyedidia/flags/flag_of_texas.png"
    #    path = "/Users/adamyedidia/flags/flag_of_uk.jpg"
#        path = "/Users/adamyedidia/flags/flag_of_us.jpeg"
#        path = "dot.png"

        imRaw = Image.open(path)

        im = np.array(imRaw.convert("RGB")).astype(float)

        batchedIm = batchAndDifferentiate(im, [(5, False), (5, False), (1, False)])
        integralIm = actOnRGB(batchedIm, integrateArray)

        viewFrame(integralIm, 5e-4)

        rotatedIm = np.pad(rotateImage(integralIm, 0, axes=(1,0)), \
            ((2, 2), (2, 2), (0, 0)), 'constant')

        viewFrame(rotatedIm, 5e-4)

#        diffIm = batchAndDifferentiate(rotatedIm, [(1, True), (1, True), (1, False)])

        diffYIm = batchAndDifferentiate(rotatedIm, [(1, True), (1, False), (1, False)])

        print np.sum(diffYIm, axis=(0,1))

        viewFrame(diffYIm, differenceImage=True, magnification=1e-2)


        diffIm = batchAndDifferentiate(diffYIm, [(1, False), (1, True), (1, False)])

        viewFrame(diffIm, differenceImage=True, magnification=1)

        print np.sum(diffIm, axis=(0,1))

    if flagMovie:
    #    imRaw = Image.open("japan_flag_garbled_new_1.png")
    #    imRaw = Image.open("texas_flag_garbled_1.png")
    #    imRaw = Image.open("texas_flag_garbled_dup_row.png")
    #    imRaw = Image.open("france_flag_garbled_1.png")
        path = "/Users/adamyedidia/flags/flag_of_texas.png"
#        path = "/Users/adamyedidia/flags/flag_of_uk.jpg"
    #    path = "/Users/adamyedidia/flags/flag_of_us.jpeg"
#        path = "/Users/adamyedidia/Desktop/adam_h.jpeg"
#        path = "dot.png"

        imRaw = Image.open(path)

        im = np.array(imRaw.convert("RGB")).astype(float)

        batchedIm = batchAndDifferentiate(im, [(1, False), (1, False), (1, False)])
        integralIm = actOnRGB(batchedIm, integrateArray)

        vNeg = np.vectorize(getNegative)

        vNegSquare = np.vectorize(getNegativeSquared)

        viewFrame(integralIm, 5e-5)

        viewFrame(batchedIm)

        for angle in range(181):
            print angle

            rotatedIm = np.pad(rotateImage(integralIm, angle, axes=(1,0)), \
                ((2, 2), (2, 2), (0, 0)), 'constant')


            diffIm = batchAndDifferentiate(rotatedIm, [(1, True), (1, True), (1, False)])



#            print vNeg(diffIm)

            print np.sum(vNegSquare(diffIm), axis=(0,1,2))
            print np.sum(vNeg(diffIm), axis=(0,1,2))
            print np.sum(diffIm, axis=(0,1,2))

#            viewFrame(vNeg(diffIm))

            viewFrame(diffIm, magnification=0.5, differenceImage=True)

            viewFrame(diffIm, differenceImage=True, filename="rotate_dot_movie/img_" + \
                padIntegerWithZeros(angle, 3) + ".png")
