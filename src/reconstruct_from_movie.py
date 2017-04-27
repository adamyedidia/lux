from __future__ import division
import numpy as np
import pylab
from math import pi, sqrt
import matplotlib.pyplot as p
from PIL import Image
from PIL import ImageFilter
import pickle
from process_image import ungarbleImageX, ungarbleImageY, \
    createGarbleMatrixX, createGarbleMatrixY, ungarbleImageXOld, \
    ungarbleImageYOld, getQ
import imageio
from video_magnifier import viewFrame, viewFrameR
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import sys

stillFrame = True
entireBatchedVideo = False

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

def xOverXSquaredPlusSigmaSquaredMaker(sigma):
    def xOverXSquaredPlusSigmaSquared(x):
        return x/(x*x + sigma*sigma)

    return xOverXSquaredPlusSigmaSquared

def ungarbleMatrixWithRegularization(Y, Uv, Sv, Vv, Uh, Sh, Vh, sigma):
    v = Y.shape[0]
    h = Y.shape[1]

    YTilde = np.dot(np.dot(np.transpose(Uv), Y), Uh)
    xIntermediate = np.dot(np.dot(Sv, YTilde), Sh)
    XTilde = np.vectorize(xOverXSquaredPlusSigmaSquaredMaker(sigma))(xIntermediate)
    X = np.dot(np.dot(np.transpose(Vv), XTilde), Vh)
    return X

def ungarbleImage(imRaw, dist, means=None, stdevs=None, raw=True, blurRadius=None, frameNum=None,
    whichUngarble="both", sigma=1):

    if raw:
        im = np.array(imRaw).astype(float)
    else:
        im = imRaw

    imAdjusted = im
    print im.shape

#    viewFrame(imAdjusted, 1, False)

    if means != None:
        imAdjusted = np.divide(im - means, np.maximum(stdevs, 0.1))
#    imAdjusted = np.divide(im, np.maximum(stdevs, 0.1))

#    print "adjusted", imAdjusted

    imAdjusted = np.flip(imAdjusted, 1)

    rearrangedIm = np.swapaxes(np.swapaxes(imAdjusted, 0, 2), 1, 2)
#    rearrangedIm = np.flip(rearrangedIm, 1)


    imageRed = rearrangedIm[0]
    imageGreen = rearrangedIm[1]
    imageBlue = rearrangedIm[2]

#    print "means", means
#    print "stdevs", stdevs

    if not blurRadius == None:
        imageRed = gaussian_filter(imageRed, blurRadius, truncate=2.)
        imageGreen = gaussian_filter(imageGreen, blurRadius, truncate=2.)
        imageBlue = gaussian_filter(imageBlue, blurRadius, truncate=2.)

    blurredImage = np.swapaxes(np.swapaxes(np.array([imageRed, imageGreen, \
        imageBlue]), 1, 2), 0, 2)

#    blurredImage = np.flip(blurredImage, 1)

#    viewFrame(blurredImage, 1, False)
#    viewFrame(blurredImage, 50, True, "frame_orig_diff" + str(frameNum).zfill(3) + ".png")
#    viewFrame(blurredImage, 10, False)

    matShape = imageRed.shape

    gmx = createGarbleMatrixX(imageRed, dist)
    gmy = createGarbleMatrixY(imageRed, dist*matShape[0]/matShape[1])

    Uv, Sv, Vv = getQ(Uv)

    if whichUngarble == "both":
#        ungarbledImageRed = ungarbleImageY(ungarbleImageX(imageRed, gmx), gmy)
#        ungarbledImageGreen = ungarbleImageY(ungarbleImageX(imageGreen, gmx), gmy)
#        ungarbledImageBlue = ungarbleImageY(ungarbleImageX(imageBlue, gmx), gmy)

        ungarbledImageRedX = ungarbleImageX(imageRed, gmx, 0)
        ungarbledImageGreenX = ungarbleImageX(imageGreen, gmx, 0)
        ungarbledImageBlueX = ungarbleImageX(imageBlue, gmx, 0)

#        blurredImageRedX = gaussian_filter(ungarbledImageRedX, 8, truncate=4.)
#        blurredImageGreenX = gaussian_filter(ungarbledImageGreenX, 8, truncate=4.)
#        blurredImageBlueX = gaussian_filter(ungarbledImageBlueX, 8, truncate=4.)

        blurredImageRedX = ungarbledImageRedX
        blurredImageGreenX = ungarbledImageGreenX
        blurredImageBlueX = ungarbledImageBlueX

#        viewFrameR(blurredImageRedX, magnification=1e-2, differenceImage=True)
#        viewFrameR(blurredImageGreenX, magnification=1e-2, differenceImage=True)
#        viewFrameR(blurredImageBlueX, magnification=1e-2, differenceImage=True)

        ungarbledImageRed = ungarbleImageY(np.flip(blurredImageRedX, 0), gmy, sigma)
        ungarbledImageGreen = ungarbleImageY(np.flip(blurredImageGreenX, 0), gmy, sigma)
        ungarbledImageBlue = ungarbleImageY(np.flip(blurredImageBlueX, 0), gmy, sigma)

#        ungarbledImageRed = ungarbleImageY(blurredImageRedX, gmy)
#        ungarbledImageGreen = ungarbleImageY(blurredImageGreenX, gmy)
#        ungarbledImageBlue = ungarbleImageY(blurredImageBlueX, gmy)

#        ungarbledImageRed = ungarbleImageY(np.flip(ungarbleImageX(imageRed, gmx), 1), gmy)
#        ungarbledImageGreen = ungarbleImageY(np.flip(ungarbleImageX(imageGreen, gmx), 1), gmy)
#        ungarbledImageBlue = ungarbleImageY(np.flip(ungarbleImageX(imageBlue, gmx), 1), gmy)

    elif whichUngarble == "x":
        ungarbledImageRed = ungarbleImageX(imageRed, gmx, sigma)
        ungarbledImageGreen = ungarbleImageX(imageGreen, gmx, sigma)
        ungarbledImageBlue = ungarbleImageX(imageBlue, gmx, sigma)

    elif whichUngarble == "y":
        ungarbledImageRed = ungarbleImageY(imageRed, gmy, sigma)
        ungarbledImageGreen = ungarbleImageY(imageGreen, gmy, sigma)
        ungarbledImageBlue = ungarbleImageY(imageBlue, gmy, sigma)

    rearrangedImage = np.array([ungarbledImageRed, ungarbledImageGreen, ungarbledImageBlue])
    reconstructedImage = np.swapaxes(np.swapaxes(rearrangedImage, 1, 2), 0, 2)

#    adjustedImage = reconstructedImage*1e-3 + np.full(shape=reconstructedImage.shape, \
#        fill_value=128.)

#    print "recon", adjustedImage.astype(np.uint8)

    return np.flip(reconstructedImage, 1)

if stillFrame:

    imRaw = Image.open("japan_flag_garbled_1.png").convert("RGB")
#    imBlurred = imRaw.filter(ImageFilter.GaussianBlur(radius=1))
    imBlurred = imRaw
#    print np.array(imRaw).shape

    #imRaw = pickle.load(open("garbled.p", "r"))
#    viewFrame(imBlurred, 1, False)

#    pylab.imshow(imBlurred)
#    p.show()

    ungarbledImage = ungarbleImage(imBlurred, 0.5, whichUngarble="both", sigma=1)
    #blurredImage = blur2DImage(ungarbledImage, 30)
    blurredImage = ungarbledImage


#    print ungarbledImage
#    print blurredImage

#    print blurredImage

    viewFrame(blurredImage, 5e1, False)

#    pylab.imshow(ungarbleImage(imBlurred, 1))
#    pylab.imshow(ungarbleImage(imBlurred, 1).filter(GaussianBlur(radius=14)))
#    p.show()

if entireBatchedVideo:
    LENGTH_OF_VIDEO = 60
    NUM_FRAMES = 1621


    print "reading pickle file..."

    video, means, stdevs = pickle.load(open("batched_pokemon_video_" + sys.argv[1] + "_" + \
        sys.argv[2] + ".p", "r"))

    startFrame = int(sys.argv[1])
    endFrame = int(sys.argv[2])

    for i, frame in enumerate(video):

        currentTime = (startFrame + (endFrame - startFrame)*i/len(video)) * \
            LENGTH_OF_VIDEO / NUM_FRAMES

        print i, "/", len(video)
        print "Video time", currentTime
#        frame = pickle.load(open("garbled.p", "r"))
#        viewFrame(frame)

#        ungarbledImage = ungarbleImage(frame, 1, means, stdevs, False, 1, i)
#        viewFrame(ungarbledImage, 1e-5, True)

        ungarbledImage = ungarbleImage(frame, 1e3, whichUngarble="both")

#        blurredImage = blur2DImage(ungarbledImage, 8)
        blurredImage = ungarbledImage

#        print blurredImage

#        viewFrame(blurredImage, 1, True, "frame_recon_blur_" + str(i).zfill(3) + ".png")
        viewFrame(blurredImage, 1e-4, False)
#        viewFrame(blurredImage, 50, False)
