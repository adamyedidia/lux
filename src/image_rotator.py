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

#def rotateImage(arr):

#    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
#    dst = cv2.warpAffine(arr,M,(cols,rows))



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
        imRaw = Image.open(path)

        im = np.array(imRaw.convert("RGB")).astype(float)

        batchedIm = batchAndDifferentiate(im, [(5, False), (5, False), (1, False)])
        integralIm = actOnRGB(batchedIm, integrateArray)

        viewFrame(integralIm)

        rotatedIm = rotateImage(integralIm, 5, axes=(1,0))

        viewFrame(rotatedIm)

        diffIm = batchAndDifferentiate(rotatedIm, [(1, True), (1, True), (1, False)])

        viewFrame(diffIm, 1, True)

    if flagMovie:
    #    imRaw = Image.open("japan_flag_garbled_new_1.png")
    #    imRaw = Image.open("texas_flag_garbled_1.png")
    #    imRaw = Image.open("texas_flag_garbled_dup_row.png")
    #    imRaw = Image.open("france_flag_garbled_1.png")
#        path = "/Users/adamyedidia/flags/flag_of_texas.png"
#        path = "/Users/adamyedidia/flags/flag_of_uk.jpg"
    #    path = "/Users/adamyedidia/flags/flag_of_us.jpeg"
        path = "/Users/adamyedidia/Desktop/adam_h.jpeg"

        imRaw = Image.open(path)

        im = np.array(imRaw.convert("RGB")).astype(float)

        batchedIm = batchAndDifferentiate(im, [(1, False), (1, False), (1, False)])
        integralIm = actOnRGB(batchedIm, integrateArray)

        viewFrame(integralIm, 5e-5)

        viewFrame(batchedIm)

        for angle in range(181):
            print angle

            rotatedIm = rotateImage(integralIm, angle, axes=(1,0))

            diffIm = batchAndDifferentiate(rotatedIm, [(1, True), (1, True), (1, False)])

            viewFrame(diffIm)

            viewFrame(diffIm, differenceImage=True, filename="rotate_pesto_movie/img_" + \
                padIntegerWithZeros(angle, 3) + ".png")
