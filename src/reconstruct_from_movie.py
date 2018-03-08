from __future__ import division
import numpy as np
import pylab
from math import pi, sqrt
import matplotlib.pyplot as p
from PIL import Image
from PIL import ImageFilter
import pickle
from process_image import ungarbleImageX, ungarbleImageY, \
    createGarbleMatrixX, createGarbleMatrixY, createGarbleMatrixFull, \
    ungarbleImageXOld, \
    ungarbleImageYOld, getQ
import imageio
from video_magnifier import viewFrame, viewFrameR
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import sys
from scipy.linalg import dft



stillFrame = False
garbleTest = False
garbleReal = True
convolveTest = False
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

def getOccluderFrameDarknessLowerLeft(shape):

    return np.array([[[255*(1-1*(max((shape[0] - i)/shape[0], j/shape[1]) < 0.5))]*3 for j in range(shape[1])] \
        for i in range(shape[0])])

def invertWithRegularization(Sv, YTilde, Sh, v, h, sigma):
    XTilde = np.zeros((v, h))

    for i in range(v):
        for j in range(h):
#            print Sv[i]*Sv[i]*Sh[j]*Sh[j], sigma*sigma, \
#                Sv[i]*Sv[i]*Sh[j]*Sh[j] + sigma*sigma

            XTilde[i][j] = Sv[i]*Sh[j]*YTilde[i][j] / \
                (Sv[i]*Sv[i]*Sh[j]*Sh[j] + sigma*sigma)

    return XTilde

def invertWithRegularizationAndSmoothing(Sv, YTilde, Sh, v, h, sigma, a):
    XTilde = np.zeros((v, h))

    for i in range(v):
#        print Sv[i]
        for j in range(h):
#            print Sv[i]*Sv[i]*Sh[j]*Sh[j], sigma*sigma, \
#                Sv[i]*Sv[i]*Sh[j]*Sh[j] + sigma*sigma

#            XTilde[i][j] = Sv[i]*Sh[j]*YTilde[i][j] / \
#                (Sv[i]*Sv[i]*Sh[j]*Sh[j] + sigma*sigma*(i/h)**a*(j/h)**a)

            XTilde[i][j] = Sv[i]*Sh[j]*YTilde[i][j] / \
                (Sv[i]*Sv[i]*Sh[j]*Sh[j] + ((sigma*(i/h))**2 + \
                (sigma*(j/h))**2)**a)

    return XTilde

def ungarbleMatrixWithRegularization(Y, Uv, Sv, Vv, Uh, Sh, Vh, sigma, a=0):
    v = Y.shape[0]
    h = Y.shape[1]

#    for i in Sv:
#        print i

#    for j in Sh:
#        print j

    YTilde = np.dot(np.dot(np.transpose(Uv), Y), Uh)

#    xIntermediate = np.dot(np.dot(Sv, YTilde), Sh)
#    print xOverXSquaredPlusSigmaSquaredMaker(sigma)(5)

#    XTilde = np.vectorize(xOverXSquaredPlusSigmaSquaredMaker(sigma))(xIntermediate)
#    XTilde = np.dot(np.dot(np.linalg.inv(Sv), YTilde), np.linalg.inv(Sh))
    XTilde = invertWithRegularizationAndSmoothing(Sv, YTilde, Sh, v, h, sigma, a)

#    print "hi", np.dot(np.dot(np.linalg.inv(Sv), YTilde), np.linalg.inv(Sh))
#    print "ho", np.vectorize(xOverXSquaredPlusSigmaSquaredMaker(sigma))(xIntermediate)

    X = np.dot(np.dot(np.transpose(Vv), XTilde), Vh)
#    for i in range(v):
#        for j in range(h):
#            print X[i][j]

    return X

def linearMagnificationFunctionMaker(magnificationFactor, maxI, maxJ):
    def linearMagnificationFunction(i, j, k):
        return 1 + magnificationFactor*((i/maxI)**2 + (j/maxJ)**2)/sqrt(2)

    return linearMagnificationFunction

def unevenlyMagnifyImage(mat, magnificationFactor):
    matShape = mat.shape

    linearMagnificationFunction = \
        linearMagnificationFunctionMaker(magnificationFactor, matShape[0], \
        matShape[1])

    magnificationArray = np.fromfunction(linearMagnificationFunction, matShape)

    return np.multiply(mat, magnificationArray)

def multiplyMatrixRGBArray(mat, rgbArray):
    rearrangedArr = np.swapaxes(np.swapaxes(rgbArray, 0, 2), 1, 2)

    redArray = rearrangedArr[0]
    greenArray = rearrangedArr[1]
    blueArray = rearrangedArr[2]

    redResult = np.dot(mat, redArray)
    greenResult = np.dot(mat, greenArray)
    blueResult = np.dot(mat, blueArray)

    multipliedImage = np.array([redResult, greenResult, blueResult])

    return np.swapaxes(np.swapaxes(multipliedImage, 1, 2), 0, 2)

def multiplyMatrixRGBArrayMatrixTranspose(mat, rgbArray, matTranspose):

    rearrangedArr = np.swapaxes(np.swapaxes(rgbArray, 0, 2), 1, 2)

    redArray = rearrangedArr[0]
    greenArray = rearrangedArr[1]
    blueArray = rearrangedArr[2]

    redResult = np.dot(np.dot(mat, redArray), np.transpose(matTranspose))
    greenResult = np.dot(np.dot(mat, greenArray), np.transpose(matTranspose))
    blueResult = np.dot(np.dot(mat, blueArray), np.transpose(matTranspose))

    multipliedImage = np.array([redResult, greenResult, blueResult])

    return np.swapaxes(np.swapaxes(multipliedImage, 1, 2), 0, 2)

def convolve2dRGB(rgb1, rgb2):

    rearranged1 = np.swapaxes(np.swapaxes(rgb1, 0, 2), 1, 2)
    rearranged2 = np.swapaxes(np.swapaxes(rgb2, 0, 2), 1, 2)

    red1, green1, blue1 = rearranged1[0], rearranged1[1], rearranged1[2]
    red2, green2, blue2 = rearranged2[0], rearranged2[1], rearranged2[2]

    convolveRed = convolve2d(red1, red2)
    convolveGreen = convolve2d(green1, green2)
    convolveBlue = convolve2d(blue1, blue2)

    convolvedImage = np.array([convolveRed, convolveGreen, convolveBlue])

    return np.swapaxes(np.swapaxes(convolvedImage, 1, 2), 0, 2)

def fftImageBothWays(im, raw=True):
    if raw:
        im = np.array(imRaw).astype(float)
    else:
        im = imRaw

    dftLeft = dft(im.shape[0])
    dftRight = dft(im.shape[1])

    return multiplyMatrixRGBArrayMatrixTranspose(dftLeft, im, dftRight)

def garbleImage(imRaw, dist, raw=True):

    if raw:
        im = np.array(imRaw).astype(float)
    else:
        im = imRaw

#    imAdjusted = np.flip(np.flip(im, 1), 0)
    imAdjusted = im
#    imAdjusted = np.flip(im, 0)

    rearrangedIm = np.swapaxes(np.swapaxes(imAdjusted, 0, 2), 1, 2)

    imageRed = rearrangedIm[0]
    imageGreen = rearrangedIm[1]
    imageBlue = rearrangedIm[2]

    matShape = imageRed.shape


    gmx = createGarbleMatrixX(imageRed, dist)
    gmy = createGarbleMatrixY(imageRed, dist*matShape[0]/matShape[1])

    garbledRed = np.dot(gmy, np.dot(imageRed, gmx))
    garbledGreen = np.dot(gmy, np.dot(imageGreen, gmx))
    garbledBlue = np.dot(gmy, np.dot(imageBlue, gmx))

    rearrangedImage = np.array([garbledRed, garbledGreen, garbledBlue])
    garbledImage = np.swapaxes(np.swapaxes(rearrangedImage, 1, 2), 0, 2)



    return np.flip(garbledImage, 0)
#    return garbledImage
#    return np.flip(np.flip(garbledImage, 1), 0)

def getUpperPartGarbleImage(im, dist, raw=True):

    if raw:
        im = np.array(imRaw).astype(float)
    else:
        im = imRaw

#    imAdjusted = np.flip(im, 0)
    imAdjusted = im

    rearrangedIm = np.swapaxes(np.swapaxes(imAdjusted, 0, 2), 1, 2)

    imageRed = rearrangedIm[0]
    imageGreen = rearrangedIm[1]
    imageBlue = rearrangedIm[2]

    matShape = imageRed.shape


    gmx = createGarbleMatrixFull(imageRed.shape[1], dist)
    gmy = createGarbleMatrixY(imageRed, dist*matShape[0]/matShape[1])

    garbledRed = np.dot(gmy, np.dot(imageRed, gmx))
    garbledGreen = np.dot(gmy, np.dot(imageGreen, gmx))
    garbledBlue = np.dot(gmy, np.dot(imageBlue, gmx))

    rearrangedImage = np.array([garbledRed, garbledGreen, garbledBlue])
    garbledImage = np.swapaxes(np.swapaxes(rearrangedImage, 1, 2), 0, 2)



#    return np.flip(garbledImage, 0)

    return garbledImage

def ungarbleImage(imRaw, dist, means=None, stdevs=None, raw=True, blurRadius=None, frameNum=None,
    whichUngarble="both", sigma=1, a=0):

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

    imAdjusted = np.flip(np.flip(imAdjusted, 1), 0)

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

    Uv, Sv, Vv = getQ(gmy, dist)
    Uh, Sh, Vh = getQ(gmx, dist*matShape[0]/matShape[1])

    print Uv.shape
    print Sv.shape
    print Vv.shape
    print matShape
    print Uh.shape
    print Sh.shape
    print Vh.shape

    if whichUngarble == "both":
        ungarbledImageRed = ungarbleMatrixWithRegularization(imageRed, \
            Uv, Sv, Vv, Uh, Sh, Vh, sigma, a)
        ungarbledImageGreen = ungarbleMatrixWithRegularization(imageGreen, \
            Uv, Sv, Vv, Uh, Sh, Vh, sigma, a)
        ungarbledImageBlue = ungarbleMatrixWithRegularization(imageBlue, \
            Uv, Sv, Vv, Uh, Sh, Vh, sigma, a)

#        ungarbledImageRed = ungarbleImageY(ungarbleImageX(imageRed, gmx), gmy)
#        ungarbledImageGreen = ungarbleImageY(ungarbleImageX(imageGreen, gmx), gmy)
#        ungarbledImageBlue = ungarbleImageY(ungarbleImageX(imageBlue, gmx), gmy)

#        ungarbledImageRedX = ungarbleImageX(imageRed, gmx, 0)
#        ungarbledImageGreenX = ungarbleImageX(imageGreen, gmx, 0) # this one
#        ungarbledImageBlueX = ungarbleImageX(imageBlue, gmx, 0)

#        blurredImageRedX = gaussian_filter(ungarbledImageRedX, 8, truncate=4.)
#        blurredImageGreenX = gaussian_filter(ungarbledImageGreenX, 8, truncate=4.)
#        blurredImageBlueX = gaussian_filter(ungarbledImageBlueX, 8, truncate=4.)

#        blurredImageRedX = ungarbledImageRedX
#        blurredImageGreenX = ungarbledImageGreenX # this one
#        blurredImageBlueX = ungarbledImageBlueX

#        viewFrameR(blurredImageRedX, magnification=1e-2, differenceImage=True)
#        viewFrameR(blurredImageGreenX, magnification=1e-2, differenceImage=True)
#        viewFrameR(blurredImageBlueX, magnification=1e-2, differenceImage=True)

#        ungarbledImageRed = ungarbleImageY(np.flip(blurredImageRedX, 0), gmy, sigma)
#        ungarbledImageGreen = ungarbleImageY(np.flip(blurredImageGreenX, 0), gmy, sigma)
#        ungarbledImageBlue = ungarbleImageY(np.flip(blurredImageBlueX, 0), gmy, sigma)
#        this one

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

#    imRaw = Image.open("japan_flag_garbled_new_1.png").convert("RGB")
    imRaw = Image.open("texas_garbled_recent_2.png")

#    imBlurred = imRaw.filter(ImageFilter.GaussianBlur(radius=2))
    imBlurred = imRaw
#    print np.array(imRaw).shape

    #imRaw = pickle.load(open("garbled.p", "r"))
#    viewFrame(imBlurred, 1, False)

#    pylab.imshow(imBlurred)
#    p.show()

    ungarbledImage = ungarbleImage(imBlurred, 0.125, whichUngarble="both", sigma=3e-5,
        a=1.)
#    blurredImage = blur2DImage(ungarbledImage, 1)
    blurredImage = ungarbledImage


#    print ungarbledImage
#    print blurredImage

#    print blurredImage

    viewFrame(blurredImage, 8e-4, False)
#    viewFrame(blurredImage, 1, False)

#    pylab.imshow(ungarbleImage(imBlurred, 1))
#    pylab.imshow(ungarbleImage(imBlurred, 1).filter(GaussianBlur(radius=14)))
#    p.show()

if convolveTest:

    path = "/Users/adamyedidia/flags/flag_of_uk.png"

    imRaw = Image.open(path)
    im = np.array(imRaw).astype(float)

    garbledImage = garbleImage(imRaw, 0.125)
    garbledImageUpperPart = getUpperPartGarbleImage(imRaw, 0.125)

    garbledImageFull = garbledImage + garbledImageUpperPart
    viewFrame(garbledImageFull, 1e3, False)

    fftIm = fftImageBothWays(imRaw)

    occ = getOccluderFrameDarknessLowerLeft(fftIm.shape)

    convolved = convolve2dRGB(im, occ)

    viewFrame(convolved)

    fftOcc = fftImageBothWays(occ)

    imOcc = np.multiply(fftIm, fftOcc)

    viewFrame(imOcc)

    goBack = fftImageBothWays(imOcc)

    viewFrame(goBack, 1, False)


if garbleReal:
#    imRaw = Image.open("japan_flag_garbled_no_left_line.png")
#    imRaw = Image.open("texas_flag_garbled_no_dot.png")
#    imRaw = Image.open("texas_flag_garbled_no_left_line.png")
#    imRaw = Image.open("texas_flag_garbled_green_square.png")
    imRaw = Image.open("texas_garbled_recent.png")

#    imRaw = Image.open("us_flag_garbled_no_line.png")

#    imRaw = Image.open("us_flag_garbled_1.png")
#    imRaw = Image.open("france_flag_garbled_no_line.png")



#    garbledImageFull = np.array(imRaw).astype(float)

    dirName = "texas_garbled"
    path = "/Users/adamyedidia/flags_garbled/" + dirName + "/downsampled.p"
    garbledImageFull = pickle.load(open(path, "r"))
#    garbledImageFull = garbledImageFull[0:1000,0:1000]

    viewFrame(garbledImageFull, 1, False)

#    ungarbledImage = ungarbleImage(garbledImageFull, 0.125, whichUngarble="both", sigma=3e-4,
#        a=1.)

    ungarbledImage = ungarbleImage(garbledImageFull, 0.125, whichUngarble="both", sigma=0,
        a=1)

    maxX = ungarbledImage.shape[0]
    maxY = ungarbledImage.shape[1]



    for i in [0, int(maxX/2), maxX-1]:
        for j in [0, int(maxY/2), maxY-1]:
            print (i, j), ungarbledImage[i][j]

#    viewFrame(ungarbledImage, 1e-2, False)

    slicedUngarbledImage = ungarbledImage[:maxX-1, 1:, :]

    blurredUngarbledImage = blur2DImage(slicedUngarbledImage, 5)

    magImage = unevenlyMagnifyImage(blurredUngarbledImage, 10)

#    viewFrame(magImage, 1e-4, False)

    newMaxX = slicedUngarbledImage.shape[0]
    newMaxY = slicedUngarbledImage.shape[1]

    for i in [0, int(newMaxX/2), newMaxX-1]:
        for j in [0, int(newMaxY/2), newMaxY-1]:
            print (i, j), slicedUngarbledImage[i][j]

    reGarbledImage = garbleImage(slicedUngarbledImage, 0.125, raw=False)
#    reGarbledImage = garbleImage(blurredUngarbledImage, 0.125, raw=False)


#    viewFrame(reGarbledImage, 1e1, False)


    ungarbledSmoothedImage = ungarbleImage(reGarbledImage, 0.125, \
        whichUngarble="both", sigma=3e-10, a=0.5)

    for i in [0, int(newMaxX/2), newMaxX-1]:
        for j in [0, int(newMaxY/2), newMaxY-1]:
            print (i, j), ungarbledSmoothedImage[i][j]

#    magImage = unevenlyMagnifyImage(ungarbledSmoothedImage, 10)
    magImage = np.flip(ungarbledSmoothedImage, 0)
#    magImage = ungarbledSmoothedImage

    viewFrame(-magImage, 5e-4, False)


if garbleTest:
    imRaw = Image.open("2000px-Flag_of_Texas.png")

    garbledImage = garbleImage(imRaw, 0.125)
    garbledImageUpperPart = getUpperPartGarbleImage(imRaw, 0.125)

    garbledImageFull = garbledImage + garbledImageUpperPart
#    garbledImageFull = garbledImage

    viewFrame(garbledImageFull, 1e3, False)

#    ungarbledImage = ungarbleImage(garbledImageFull, 0.125, whichUngarble="both", sigma=3e-4,
#        a=1.)

    ungarbledImage = ungarbleImage(garbledImageFull, 0.125, whichUngarble="both", sigma=0,
        a=1.)

    maxX = ungarbledImage.shape[0]
    maxY = ungarbledImage.shape[1]



    for i in [0, int(maxX/2), maxX-1]:
        for j in [0, int(maxY/2), maxY-1]:
            print (i, j), ungarbledImage[i][j]

    viewFrame(ungarbledImage, 1, False)

    slicedUngarbledImage = ungarbledImage[:maxX-1, 1:, :]

    newMaxX = slicedUngarbledImage.shape[0]
    newMaxY = slicedUngarbledImage.shape[1]

    for i in [0, int(newMaxX/2), newMaxX-1]:
        for j in [0, int(newMaxY/2), newMaxY-1]:
            print (i, j), slicedUngarbledImage[i][j]

    reGarbledImage = garbleImage(slicedUngarbledImage, 0.125, raw=False)

    viewFrame(reGarbledImage, 1e3, False)

    ungarbledSmoothedImage = ungarbleImage(reGarbledImage, 0.125, \
        whichUngarble="both", sigma=3e-4, a=1.)

    for i in [0, int(newMaxX/2), newMaxX-1]:
        for j in [0, int(newMaxY/2), newMaxY-1]:
            print (i, j), ungarbledSmoothedImage[i][j]

    viewFrame(ungarbledSmoothedImage, 1, False)


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
