import random
import pickle
import matplotlib.pyplot as p
import numpy as np
from math import sin, exp, log, cos, sin, tan, atan, \
    atan2, floor, ceil, sqrt, pi
import sys
import os
from video_processor import padIntegerWithZeros
from video_magnifier import viewFrame
from image_distortion_simulator import batch
from scipy.signal import convolve2d, convolve
from image_distortion_simulator import doFuncToEachChannel, \
    doFuncToEachChannelTwoInputs, imageify, separate, doFuncToEachChannelTwoInputsVec

from scipy.optimize import least_squares



BASIC_CONVOLVE = False
SCALE_CONVOLVE = False
SCALE_PLUS_SHIFT_CONVOLVE = False
TAN_SHIFT = False
CAMERA_SIM_1D = False
CAMERA_SIM_1D_RECOVERY = False
RECOVER_SHIFT_AND_ZOOM_2D = False
OPTICAL_FLOW_PARTICLE = True


def instantMeanSubFrame(frame):
    numPixels = frame.shape[0]*frame.shape[1]
    averagePixel = np.sum(np.sum(frame, 0), 0)/numPixels

    frameDims = frame.shape[:-1]
    meanPixelFrame = np.array([[averagePixel for i in range(frameDims[1])] for j in range(frameDims[0])])

    return frame - meanPixelFrame

def imPoly(x):
	return abs(x+1e-10)**1j

def imPolyReal(x):
	return np.real(imPoly(x))

def xexp(x):
	return x*exp(-abs(x))

def sign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	elif x == 0:
		return 0

def getLoggedValsMatrix(plausibleZeroes, exponentialWeightFactor, \
	modifiedF):

	posMatrix = []
	negMatrix = []

	for zero in plausibleZeroes:
		posMatrix.append([modifiedF(zero+exp(exponentialWeightFactor*(x))) for x in xs])
		negMatrix.append([modifiedF(zero-exp(exponentialWeightFactor*(x))) for x in xs])

	return posMatrix, negMatrix

def getShift(vals1, val2):
	fftVals1 = np.fft.fft(np.concatenate([vals1, np.zeros(numPoints-1)]))
	fftVals2 = np.fft.fft(np.concatenate([vals2[::-1], np.zeros(numPoints-1)]))

def getOffset(f1, f2, thetas, minTheta, maxTheta, numPoints):
#	shiftAmount = 20


	vals = [f1(atan(theta)) for theta in thetas]
	valsShifted = [f2(atan(theta)) for theta in thetas]

	print(xs)

	p.clf()
	p.plot(thetas, vals)
	p.plot(thetas, valsShifted)	
	p.show()

	fftVals = np.fft.fft(np.concatenate([vals, np.zeros(numPoints-1)]))
	fftValsShifted = np.fft.fft(np.concatenate([valsShifted[::-1], np.zeros(numPoints-1)]))

	p.plot(np.real(fftVals))
	p.plot(np.real(fftValsShifted))
	p.show()

	fftValsMultiplied = np.multiply(fftVals, fftValsShifted)

	p.plot(np.real(np.multiply(fftVals, fftValsShifted)))
	p.show()

#	p.plot(np.linspace(-200, 200, 1999), np.convolve(vals, valsShifted))
	p.plot(np.linspace(2*minTheta, 2*maxTheta, 2*numPoints-1), np.fft.ifft(fftValsMultiplied))
#	p.axvline(x=shiftAmount)
	p.show()

def linearResultFunc2d(x, t):     
    return x[0]*t[0] + x[1]*t[1] + x[2]

def fuzzyLookup2D(array, i, j):
	arrShape = array.shape

	if i < 0:
		i = 0
	if i > arrShape[0] - 1:
		i = arrShape[0] - 1

	if j < 0:
		j = 0
	if j > arrShape[1] - 1:
		j = arrShape[1] - 1

	floorI = int(floor(i))
	ceilI = int(ceil(i))
	floorJ = int(floor(j))
	ceilJ = int(ceil(j))
    
	resI = i % 1
	resJ = j % 1
    
	fIfJ = array[floorI][floorJ]
	fIcJ = array[floorI][ceilJ]
	cIfJ = array[ceilI][floorJ]
	cIcJ = array[ceilI][ceilJ]

#    print fIfJ, fIcJ, cIfJ, cIcJ
    
	return (1-resI)*(1-resJ)*fIfJ + \
		(1-resI)*resJ*fIcJ + \
		resI*(1-resJ)*cIfJ + \
		resI*resJ*cIcJ

def downsample(im, downsampleFactorX, downsampleFactorY):
    imShape = im.shape

    indexX = (downsampleFactorX - 1)/2

    returnArray = []

    while indexX < imShape[0]:
        returnArray.append([])
        indexY = (downsampleFactorY - 1)/2

        while indexY < imShape[1]:
            returnArray[-1].append(fuzzyLookup2D(im, indexX, indexY))
            indexY += downsampleFactorY

        indexX += downsampleFactorX


    return np.array(returnArray)

def upsample(im, upsampleFactorX, upsampleFactorY):
    imShape = im.shape

    xs = np.linspace(-0.5+1/(2*upsampleFactorX), \
        imShape[0]-0.5-1/(2*upsampleFactorX), imShape[0]*upsampleFactorX)

    ys = np.linspace(-0.5+1/(2*upsampleFactorY), \
        imShape[1]-0.5-1/(2*upsampleFactorY), imShape[1]*upsampleFactorY)
    

#    print(xs)

    returnArr = []

    for x in xs:
        returnArr.append([])
        for y in ys:
            returnArr[-1].append(fuzzyLookup2D(im, x, y))

    return np.array(returnArr)

def shiftWraparound(im, horizontalShift, verticalShift):
	shiftedIm = np.roll(np.roll(im, verticalShift, axis=0), horizontalShift, axis=1)

	return shiftedIm

def cropAboutPoint(im, x, y, xShape, yShape):
#    print(x, y, xShape, yShape)

#    print(x-int(xShape/2),x+xShape-int(xShape/2), \
#        y-int(yShape/2),y+yShape-int(yShape/2))

    return im[x-int(xShape/2):x+xShape-int(xShape/2), \
        y-int(yShape/2):y+yShape-int(yShape/2)]

def convolveImagesValid(im1, im2):
    return doFuncToEachChannelTwoInputs(lambda x, y: convolve2d(x, y, "valid"), \
        im1, im2)

def convolveImagesValidOneDimension(im1, im2):
#    for vec1, vec2 in zip(im1, im2):
#        p.plot([i[0] for i in vec1])
#        p.plot([i[0] for i in vec2])
 #       p.show()

    return np.array([doFuncToEachChannelTwoInputsVec(lambda x, y: convolve(x, y, "valid"), \
        vec1, vec2) for vec1, vec2 in zip(im1, im2)])

def colorFlatten(im):
    return np.sum(im, 2)/3

def getShiftBetweenCroppedImages(im1, im2, granularity):
    croppedDora = im1
    croppedShiftedDora = im2

    croppedUpsampledDora = upsample(croppedDora, granularity, granularity)
    croppedUpsampledShiftedDora = upsample(croppedShiftedDora, \
        granularity, granularity)

    convolveResult = convolveImagesValid(np.flip(np.flip(croppedUpsampledDora, 0), 1), croppedUpsampledShiftedDora)
    colorFlatResult = colorFlatten(convolveResult)

    maxIndex = np.unravel_index(np.argmax(colorFlatResult, axis=None), \
        colorFlatResult.shape)

    convolveResultShape = convolveResult.shape
    convolveResultCenterX = int((convolveResultShape[0]-1)/2)
    convolveResultCenterY = int((convolveResultShape[1]-1)/2)

    return maxIndex - np.array([convolveResultCenterX, convolveResultCenterY])

def getSineWaveBetweenTransformedImages(im1, im2):
    print(im1.shape)
    print(im2.shape)

    convolveResult = convolveImagesValidOneDimension(np.flip(im1, 1), \
       im2)

    argmaxes = [np.amax(vec) for vec in convolveResult]

    p.plot(argmaxes)
    p.show()

    colorFlatResult = colorFlatten(convolveResult)

#    print(colorFlatResult)

    viewFrame(imageify(np.transpose(colorFlatResult)), adaptiveScaling=True)

def getScalingBetweenTransformedImages(im1, im2):
    convolveResult = convolveImagesValid(np.flip(np.flip(im1, 0), 1), im2)
    colorFlatResult = colorFlatten(convolveResult)

#    print(colorFlatResult)

#    viewFrame(imageify(colorFlatResult), adaptiveScaling=True)
    maxIndex = np.unravel_index(np.argmax(colorFlatResult, axis=None), \
        colorFlatResult.shape)

    convolveResultShape = convolveResult.shape

    convolveResultCenterX = int((convolveResultShape[0]-1)/2)
    convolveResultCenterY = int((convolveResultShape[1]-1)/2)    

    return maxIndex - np.array([convolveResultCenterX, convolveResultCenterY])

def exponentialTransform(im, xs, ys, attenuationFactor):

    returnArray = []

    imShape = im.shape

    imCenterX = (imShape[0]-1)/2
    imCenterY = (imShape[1]-1)/2

    thetas = []

    for x in xs:   
        returnArray.append([])
#        adjustedX = x - imCenterX

        for y in ys:
#            adjustedY = y - imCenterY
            theta = atan2(y, x)
            r = sqrt(x**2 + y**2)

            indexX = imCenterX + 0.3*cos(theta)*exp(attenuationFactor*r)
            indexY = imCenterY + 0.3*sin(theta)*exp(attenuationFactor*r)

#            thetas.append(theta)
#            p.plot(theta, sin(theta))
            p.plot(indexX, indexY, "bo")

#            print(x, y, theta, r, cos(theta), sin(theta), indexX, indexY)

            val = fuzzyLookup2D(im, indexX, indexY)

            returnArray[-1].append(val)

#    p.plot(thetas)
    p.show()

    return np.array(returnArray)


def polarTransform(im, focusX, focusY, rs, thetas):

    returnArray = []

    for r in rs:
        returnArray.append([])    
        for theta in thetas:

            indexX = focusX + cos(theta)*r
            indexY = focusY + sin(theta)*r

#            p.plot(indexX, indexY, "bo")

            val = fuzzyLookup2D(im, indexX, indexY)

            returnArray[-1].append(val)            

#    p.show()

    return np.array(returnArray)

def exponentialTransformRTheta(im, focusX, focusY, logRs, thetas):

    returnArray = []

    for logR in logRs:
        returnArray.append([])    
        for theta in thetas:

            indexX = focusX + cos(theta)*exp(logR)
            indexY = focusY + sin(theta)*exp(logR)

#            p.plot(indexX, indexY, "bo")

            val = fuzzyLookup2D(im, indexX, indexY)

            returnArray[-1].append(val)            

#    p.show()

    return np.array(returnArray)

def gaussianImageKernel(maxX, maxY, sigmasPerSidelength):
    centerX = (maxX-1)/2
    centerY = (maxY-1)/2

    returnArray = []

    for x in range(maxX):
        returnArray.append([])
        for y in range(maxY):
            distance = sqrt((x - centerX)**2 + (y - centerY)**2)

            val = exp(-2*distance*sigmasPerSidelength/maxX)
#            print(val)

            returnArray[-1].append([val]*3)


#    viewFrame(np.array(returnArray), adaptiveScaling=True)
    return np.array(returnArray)

def getRotationBetweenTransformedImages(im1, im2):
    convolveResult = convolveImagesValid(np.flip(np.flip(im1, 0), 1), im2)
    colorFlatResult = colorFlatten(convolveResult)

#    print(colorFlatResult)

#    viewFrame(imageify(colorFlatResult), adaptiveScaling=True)
    maxIndex = np.unravel_index(np.argmax(colorFlatResult, axis=None), \
        colorFlatResult.shape)

    convolveResultShape = convolveResult.shape

    convolveResultCenterX = int((convolveResultShape[0]-1)/2)
    convolveResultCenterY = int((convolveResultShape[1]-1)/2)    

    return maxIndex - np.array([convolveResultCenterX, convolveResultCenterY])

def getScaleAndShiftAtFocus(im1, im2, focusX, focusY, cropX, cropY, granularity):

    dora = im1
    scaledDora = im2

    croppedDora = instantMeanSubFrame(cropAboutPoint(dora, focusX, focusY, cropX, cropY))
    croppedDora = np.multiply(croppedDora, gaussianImageKernel(croppedDora.shape[0], \
        croppedDora.shape[1], 2))
    croppedScaledDora = instantMeanSubFrame(cropAboutPoint(scaledDora, focusX, focusY, cropX*2+1, cropY*2+1))

    miniCroppedScaledDora = instantMeanSubFrame(cropAboutPoint(scaledDora, focusX, focusY, cropX, cropY))

#    viewFrame(upsample(croppedDora, granularity, granularity), \
#        differenceImage=True, adaptiveScaling=True)
#    viewFrame(upsample(miniCroppedScaledDora, granularity, granularity), \
#        differenceImage=True, adaptiveScaling=True)

    shift = getShiftBetweenCroppedImages(croppedDora, croppedScaledDora, granularity)

    print(shift)

    geometricGranularity = 100
    angularGranularity = 100
    radialGranularity = 100

    rs = np.linspace(1, cropX, radialGranularity)

    logRs = np.linspace(0.5, 2.5, geometricGranularity)
    logRsExtended = np.linspace(-0.5, 3.5, 2*geometricGranularity-1)

#    logRs = np.linspace(1.25, 2.75, 25)
#    logRsExtended = np.linspace(0.5, 3.5, 49)
    thetas = np.linspace(pi, 3*pi, angularGranularity)
    thetasExtended = np.linspace(0, 4*pi, 2*angularGranularity-1)

#    print(focusX, focusY)
#    print(focusX + shift[0]/granularity, focusY + shift[1]/granularity)


#    transformed1 = exponentialTransformRTheta(im1, focusX, focusY, logRs, thetas)
#    transformed2 = exponentialTransformRTheta(im2, focusX + shift[0]/granularity, focusY + shift[1]/granularity, \
#        logRsExtended, thetasExtended)

#    miniTransformed2 = exponentialTransformRTheta(im2, focusX + shift[0]/granularity, focusY + shift[1]/granularity, \
#        logRs, thetas)

    transformed1 = polarTransform(im1, focusX, focusY, rs, thetas)
    transformed2 = polarTransform(im2, focusX, focusY, rs, thetasExtended)

#    viewFrame(transformed1)
#    viewFrame(transformed2)

    rotation = getRotationBetweenTransformedImages(instantMeanSubFrame(transformed1), \
        instantMeanSubFrame(transformed2))
    rotation = rotation[1]

    print("rotation", rotation)
    rotation=0

    thetasShifted = np.linspace(pi + rotation, 3*pi + rotation, angularGranularity)

    transformed1 = instantMeanSubFrame(exponentialTransformRTheta(im1, focusX, focusY, logRs, thetas))
    transformed2 = instantMeanSubFrame(exponentialTransformRTheta(im2, focusX, focusY, logRsExtended, thetasShifted))

    viewFrame(transformed1)
    viewFrame(transformed2)

    sineShift = getSineWaveBetweenTransformedImages(np.swapaxes(transformed1,0,1), \
        np.swapaxes(transformed2,0,1))

    scalingIndices = [0,0]

#    print(scalingIndices)

    return shift[0]/granularity, shift[1]/granularity, \
        exp(2*scalingIndices[0]/geometricGranularity), scalingIndices[1]*2*pi/angularGranularity

def getDiffAtFocus(im1, im2, focusX, focusY, cropX, cropY, granularity):

    dora = im1
    scaledDora = im2

    croppedDora = instantMeanSubFrame(cropAboutPoint(dora, focusX, focusY, cropX, cropY))
    croppedScaledDora = instantMeanSubFrame(cropAboutPoint(scaledDora, focusX, focusY, cropX*2+1, cropY*2+1))

    miniCroppedScaledDora = instantMeanSubFrame(cropAboutPoint(scaledDora, focusX, focusY, cropX, cropY))

#    viewFrame(upsample(croppedDora, granularity, granularity))
#    viewFrame(upsample(miniCroppedScaledDora, granularity, granularity))

    shift = getShiftBetweenCroppedImages(croppedDora, croppedScaledDora, granularity)

#    print(shift)

    return fuzzyLookup2D(im1, focusX, focusY) - \
        fuzzyLookup2D(im2, focusX + shift[0]/granularity, focusY + shift[1]/granularity)

def getDiffGivenFocusAndParams(im1, im2, x, y, focusX, focusY, shiftX, shiftY, scaleFactor, rotation):
    proxyX = x + shiftX + (x - focusX)*(scaleFactor - 1)
    proxyY = y + shiftY + (y - focusY)*(scaleFactor - 1)

#    print(x, y, proxyX, proxyY)

    return fuzzyLookup2D(im1, x, y) - fuzzyLookup2D(im2, proxyX, proxyY)






def microconvDiff(im1, im2, cropX, cropY, granularity):

    imShape = im1.shape

    returnArray = []

    for x in range(cropX+1, imShape[0]-cropX-1):
        print(x, "/", imShape[0]-cropX-1)

        returnArray.append([])
        for y in range(cropY+1, imShape[1]-cropY-1):
            diff = getScaleAndShiftAtFocus(im1, im2, x, y, cropX, cropY, granularity)
#            print(diff)
            returnArray[-1].append(diff)


    return np.array(returnArray)

def microconvDiffWithScaling(im1, im2, cropX, cropY, granularity):

    imShape = im1.shape

    returnArray = np.zeros(imShape)

    for focusX in range(cropX+1, imShape[0]-cropX-1, cropX):
        for focusY in range(cropY+1, imShape[1]-cropY-1, cropY):

            print(focusX, focusY)

            shiftX, shiftY, scaling, rotation = \
                getScaleAndShiftAtFocus(im1, im2, focusX, focusY, cropX, cropY, granularity)

#            print(scaling)
#            scaling=1

            for x in range(focusX-int(cropX/2), focusX+cropX-int(cropX/2)):
                for y in range(focusY-int(cropY/2), focusY+cropY-int(cropY/2)):

                    diff = getDiffGivenFocusAndParams(im1, im2, x, y, \
                        focusX, focusY, shiftX, shiftY, scaling, rotation)

                    returnArray[x][y] = diff

#            print(diff)

#                    returnArray[-1].append(diff)


    return np.array(returnArray)    

#def getShiftAndZoom2D(arr):


def takeSophisticatedDifferenceFrame(frame, prevFrame, \
    magHammer, angHammer, batchFactor):

    frameShape = frame.shape

    returnArray = []

    for i in range(frameShape[0]):
        returnArray.append([])
        for j in range(frameShape[1]):
            currentPixel = frame[i][j]
            mag = linearResultFunc2d(magHammer, \
                np.array([i/batchFactor,j/batchFactor]))/batchFactor
            ang = linearResultFunc2d(angHammer, \
                np.array([i/batchFactor,j/batchFactor]))

#            prevI = i + sin(ang)*mag
#            prevJ = j + cos(ang)*mag

            prevI = i - sin(ang)*mag
            prevJ = j - cos(ang)*mag
#            prevI = i
#            prevJ = j

#            print("cos", cos(ang)*mag)
#            print("sin", sin(ang)*mag)


            prevPixel = fuzzyLookup2D(prevFrame, prevI, prevJ)
            returnArray[-1].append(currentPixel - prevPixel)

    return np.array(returnArray)

if BASIC_CONVOLVE:
	minVal = -30
	maxVal = 30
	numPoints = 1000

	xs = np.linspace(minVal, maxVal, numPoints)

	shiftAmount = 20


	vals = [xexp(x) for x in xs]
	valsShifted = [xexp(x+shiftAmount) for x in xs]

	p.plot(xs, vals)
	p.plot(xs, valsShifted)	
	p.show()

	fftVals = np.fft.fft(np.concatenate([vals, np.zeros(numPoints-1)]))
	fftValsShifted = np.fft.fft(np.concatenate([valsShifted[::-1], np.zeros(numPoints-1)]))

	p.plot(np.real(fftVals))
	p.plot(np.real(fftValsShifted))
	p.show()

	fftValsMultiplied = np.multiply(fftVals, fftValsShifted)

	p.plot(np.real(np.multiply(fftVals, fftValsShifted)))
	p.show()

#	p.plot(np.linspace(-200, 200, 1999), np.convolve(vals, valsShifted))
	p.plot(np.linspace(2*minVal, 2*maxVal, 2*numPoints-1), np.fft.ifft(fftValsMultiplied))
	p.axvline(x=shiftAmount)
	p.show()


#	p.plot(np.real(np.fft.ifft(fftValsMultiplied)))
#	p.show()


if SCALE_CONVOLVE:
	minVal = -30
	maxVal = 30
	numPoints = 1000

#	f = lambda x: sin(0.5*x)

	extraFactor = 0.6

	f = xexp
	scaledF = lambda x: f(extraFactor*x)


	xs = np.linspace(minVal, maxVal, numPoints)

	vals = [f(x) for x in xs]
	valsScaled = [scaledF(x) for x in xs]

	p.plot(xs, vals)
	p.plot(xs, valsScaled)	
	p.show()

#	rescaledXs = [sign(x)*exp(0.1*x) for x in xs]

	loggedVals = [f(exp(0.1*x)) for x in xs]
	loggedValsScaled = [scaledF(exp(0.1*x)) for x in xs]

	p.plot(xs, loggedVals)
	p.plot(xs, loggedValsScaled)	
	p.show()

	fftVals = np.fft.fft(np.concatenate([loggedVals, np.zeros(numPoints-1)]))
	fftValsScaled = np.fft.fft(np.concatenate([loggedValsScaled[::-1], np.zeros(numPoints-1)]))

	p.plot(np.real(fftVals))
	p.plot(np.real(fftValsScaled))
	p.show()

	fftValsMultiplied = np.multiply(fftVals, fftValsScaled)

	p.plot(np.real(fftValsMultiplied))
	p.show()



	print(np.dot(loggedVals, loggedValsScaled))

#	p.plot(np.convolve(loggedVals, loggedValsScaled[::-1]))
#	p.show()

	p.plot(np.linspace(2*minVal, 2*maxVal, 2*numPoints-1), np.convolve(loggedVals, loggedValsScaled[::-1]))
	p.plot(np.linspace(2*minVal, 2*maxVal, 2*numPoints-1), np.fft.ifft(fftValsMultiplied))
	p.axvline(x=10*log(extraFactor))
	p.show()

	maxIndex = np.argmax(np.fft.ifft(fftValsMultiplied))
	print(maxIndex)

	extraFactor = exp(0.1*np.linspace(2*minVal, 2*maxVal, 2*numPoints-1)[maxIndex])

	print(extraFactor)

if SCALE_PLUS_SHIFT_CONVOLVE:
	minVal = -30
	maxVal = 30
	numPoints = 300

#	f = lambda x: sin(0.5*x)

	extraFactor = 0.3
	shift = 1.0

	f = xexp
	modifiedF = lambda x: f(extraFactor*x + shift)


	xs = np.linspace(minVal, maxVal, numPoints)

	vals = [f(x) for x in xs]
	valsScaled = [modifiedF(x) for x in xs]

	p.plot(xs, vals)
	p.plot(xs, valsScaled)	
	p.show()

#	rescaledXs = [sign(x)*exp(0.1*x) for x in xs]

#	loggedVals = [f(exp(0.1*x)) for x in xs]
#	loggedValsScaled = [modifiedF(exp(0.1*x)) for x in xs]

	plausibleZeroes = np.linspace(-5, 5, numPoints)
	exponentialWeightFactor = 0.1

	posMatrix, negMatrix = getLoggedValsMatrix(plausibleZeroes, exponentialWeightFactor, modifiedF)

	posLoggedVals = [f(exp(0.1*x)) for x in xs]
	negLoggedVals = [f(-exp(0.1*x)) for x in xs]

	p.matshow(posMatrix)
	p.show()
	p.matshow(negMatrix)
	p.show()

	fftPosMat = [np.fft.fft(np.concatenate([i, np.zeros(numPoints-1)])) for i in posMatrix]
	fftNegMat = [np.fft.fft(np.concatenate([i, np.zeros(numPoints-1)])) for i in negMatrix]

	fftPosVec = np.fft.fft(np.concatenate([posLoggedVals[::-1], np.zeros(numPoints-1)]))
	fftNegVec = np.fft.fft(np.concatenate([negLoggedVals[::-1], np.zeros(numPoints-1)]))

	fftPosMultMat = [np.multiply(i, fftPosVec) for i in fftPosMat]
	fftNegMultMat = [np.multiply(i, fftNegVec) for i in fftNegMat]

	convPosMat = [np.fft.ifft(i) for i in fftPosMultMat]
	convNegMat = [np.fft.ifft(i) for i in fftNegMultMat]

	combinedConvMat = np.array(convPosMat) + np.array(convNegMat)

	print(np.array(combinedConvMat).shape)

	p.matshow(np.array(combinedConvMat).astype(float))
	p.colorbar()
	p.show()

	maxIndex = np.unravel_index(np.argmax(combinedConvMat, axis=None), \
		combinedConvMat.shape)

	print(maxIndex)
	recoveredFactor = exp(-exponentialWeightFactor*\
		np.linspace(2*minVal, 2*maxVal, 2*numPoints-1)[maxIndex[1]])

	recoveredShift = -plausibleZeroes[maxIndex[0]]

	print(recoveredFactor)
	print(recoveredShift*recoveredFactor)

if TAN_SHIFT:

	minTheta = -1.2
	maxTheta = 1.2
	numPoints = 300

	y = 3
#	f = sin
	f = xexp

	shift = 0

	tannedF = lambda theta: f(y*tan(theta))
	tannedShiftedF = lambda theta: f(5*tan(theta) + shift)

	thetas = np.linspace(minTheta, maxTheta, numPoints)
	tanThetas = np.linspace(tan(minTheta), tan(maxTheta), numPoints)

	vals = [tannedF(theta) for theta in thetas]
	shiftedVals = [tannedShiftedF(theta) for theta in thetas]

	p.plot(vals)
	p.plot(shiftedVals)
	p.show()

	moddedVals = [tannedF(atan(tanTheta)) for tanTheta in tanThetas]
	moddedShiftedVals = [tannedShiftedF(atan(tanTheta)) for tanTheta in tanThetas]

	p.plot(moddedVals)
	p.plot(moddedShiftedVals)
	p.show()

if CAMERA_SIM_1D:

	minTheta = -1.2
	maxTheta = 1.2
	numPoints = 300

	startingY = 3
	startingTilt = 0
	startingX = 0

	thetas = np.linspace(minTheta, maxTheta, numPoints)

	f = xexp

	maxT = 100

	y = startingY
	tilt = startingTilt
	x = startingX

	listOfObservations = []
	listOfFs = []

	ys = []
	tilts = []
	xs = []

	for t in range(maxT):
		print(t)

		modifiedF = lambda theta: f(y*tan(theta+tilt)+x)
		observation = [f(y*tan(theta+tilt)+x) for theta in thetas]

		ys.append(y)
		tilts.append(tilt)
		xs.append(x)
		listOfObservations.append(observation)
		listOfFs.append(modifiedF)

		y += np.random.normal(0,0.0)
		tilt += np.random.normal(0,0.00)
		x += np.random.normal(0,0.1)

		p.clf()
		p.plot(observation)
		p.savefig("zoom_sim_frames/frame_" + padIntegerWithZeros(t, 3) + ".png")

	os.system("ffmpeg -r 10 -f image2 -s 500x500 " + \
    	"-i zoom_sim_frames/frame_%03d.png " + \
		"-vcodec libx264 -crf 25 -pix_fmt yuv420p zoom_sim_vid.mp4")

	pickle.dump([listOfObservations, ys, tilts, xs], open("zoom_sim_package.p", "wb"))

	p.clf()
	p.plot(range(maxT), ys, label="ys")
	p.plot(range(maxT), tilts, label="tilts")
	p.plot(range(maxT), xs, label="xs")
	p.legend()
	p.show()

if CAMERA_SIM_1D_RECOVERY:
	[listOfObservations, ys, tilts, xs] = pickle.load(open("zoom_sim_package.p", "rb"))

	t1 = 3
	t2 = 4

	minTheta = -1.2
	maxTheta = 1.2
	numPoints = 300

	thetas = np.linspace(minTheta, maxTheta, numPoints)

	f = xexp

	y1, y2 = ys[t1], ys[t2]
	tilt1, tilt2 = tilts[t1], tilts[t2]
	x1, x2 = xs[t1], xs[t2]


	thetas = np.linspace(minTheta, maxTheta, numPoints)
	tanThetas = np.linspace(tan(minTheta), tan(maxTheta), numPoints)

	modifiedF1 = lambda theta: f(y1*tan(theta+tilt1)+x1)
	modifiedF2 = lambda theta: f(y2*tan(theta+tilt2)+x2)

	offset = getOffset(modifiedF1, modifiedF2, thetas, minTheta, maxTheta, \
		numPoints)






#	vals1 = [modifiedF1(theta) for theta in thetas]
#	vals2 = [modifiedF2(theta) for theta in thetas]

#	p.plot(vals)
#	p.plot(shiftedVals)
#	p.show()

#	moddedVals1 = [tannedF(atan(tanTheta)) for tanTheta in tanThetas]
#	moddedVals2 = [tannedShiftedF(atan(tanTheta)) for tanTheta in tanThetas]

#	p.plot(moddedVals)
#	p.plot(moddedShiftedVals)
#	p.show()



	firstObservation = listOfObservations[t1]
	secondObservation = listOfObservations[t2]



	p.plot(firstObservation)
	p.plot(secondObservation)
	p.show()

if RECOVER_SHIFT_AND_ZOOM_2D:
    dora = pickle.load(open("dora_very_downsampled_python3friendly.p", "rb"))

#    viewFrame(im)
#    viewFrame(upsample(im, 5, 7))
#    viewFrame(downsample(upsample(im, 5, 7), 5, 7))

    doraShape = dora.shape

    centerX = int((doraShape[0]-1)/2)
    centerY = int((doraShape[1]-1)/2)

    granularity = 8
    horizontalShift = 13
    verticalShift = 9

    scalingFactorX = 1.07
    scalingFactorY = 1.1

    cropX = 9
    cropY = 9

    upsampledDora = upsample(dora, granularity, granularity)
    shiftedUpsampledDora = shiftWraparound(upsampledDora, horizontalShift, verticalShift)
    shiftedDora = downsample(shiftedUpsampledDora, granularity, granularity)

 #   viewFrame(dora)
 #   viewFrame(shiftedDora)

    croppedDora = instantMeanSubFrame(cropAboutPoint(dora, centerX, centerY, cropX, cropY))
    croppedShiftedDora = instantMeanSubFrame(cropAboutPoint(shiftedDora, centerX, centerY, cropX*2+1, cropY*2+1))

#    print(getShiftBetweenCroppedImages(croppedDora, croppedShiftedDora))

    scaledDora = cropAboutPoint(upsample(dora, scalingFactorX, scalingFactorY), \
        int(centerX*scalingFactorX), int(centerY*scalingFactorY), doraShape[0], doraShape[1])

    viewFrame(dora)
    viewFrame(scaledDora)

#    croppedDora = instantMeanSubFrame(cropAboutPoint(dora, centerX, centerY, cropX, cropY))
#    croppedScaledDora = instantMeanSubFrame(cropAboutPoint(scaledDora, centerX, centerY, cropX*2+1, cropY*2+1))

#    viewFrame(croppedDora)
#    viewFrame(croppedScaledDora)

#    print(getShiftBetweenCroppedImages(croppedDora, croppedScaledDora, granularity))


#    viewFrame(dora - scaledDora, differenceImage=True, adaptiveScaling=True, magnification=2)

    diffResult = microconvDiffWithScaling(dora, scaledDora, cropX, cropY, granularity)

#    print(np.sum(np.abs(diffResult)))
    viewFrame(diffResult, \
        differenceImage=True, adaptiveScaling=True, magnification=2)

#    for x in range(cropX+1, doraShape[0]-cropX-1):
#        for y in range(cropY+1, doraShape[1]-cropY-1):
#            print(x,y,getScaleAndShiftAtFocus(dora, scaledDora, x, y, cropX, cropY, granularity))


    #print(getScaleAndShiftAtFocus(dora, scaledDora, 15, 27, cropX, cropY, granularity))



#    xs = np.linspace(-14, 14, 21)
#    ys = np.linspace(-14, 14, 21)

#    logRs = np.linspace(-1, 3, 50)
#    thetas = np.linspace(0, 2*pi, 50)



#    transformedDora = exponentialTransform(dora, xs, ys, 0.2)
#    transformedDora = exponentialTransformRTheta(dora, logRs, thetas)
#    p.imshow(dora/255)
#    p.show()
#    transformedScaledDora = exponentialTransform(scaledDora, xs, ys, 0.2)
#    transformedScaledDora = exponentialTransformRTheta(scaledDora, logRs, thetas)

#    viewFrame(transformedDora)
#    viewFrame(transformedScaledDora)




#    viewFrame(cropAboutPoint(im, 30, 30, 5, 5))



if OPTICAL_FLOW_PARTICLE:

    mags, angs = pickle.load(open("sidewalk.p", "rb"))
    vid = pickle.load(open("sidewalk_vid.p", "rb"))

    print(len(vid))
    print(len(mags))

#    print(vid.shape)

    magDict = {}

    firstMag = mags
    firstAng = angs

#    print(firstMag.shape)
    magShape = mags[0].shape
#    print(magShape)
#    print(angs[0].shape)

    THRESHOLD = 0.1

    def linearFunc(x, t, y):
        return x[0]*t + x[1] - y

    def linearFunc2d(x, t, y):
#        print(x.shape)
#        print(t.shape)
        return x[0]*t[0] + x[1]*t[1] + x[2] - y

    def linearResultFunc(x, t):
        return x[0]*t + x[1]   

    def linearResultFunc2d(x, t):     
        return x[0]*t[0] + x[1]*t[1] + x[2]

    def generateData(func, x, ts):
        return np.array([func(x, t) for t in ts])


    prevFrame = vid[0]


    for t in range(1, len(mags)):

        frame = vid[t]

#        viewFrame(imageify(mags[t]), adaptiveScaling=True)

        mag = mags[t-1]
        ang = angs[t-1]



#        for i in range(magShape[0]):
#            print(i)

#            for j in range(magShape[1]):
#                singleMag = mag[i][j]

        mag_train = []
        ang_train = []
        t_train = []
        x0 = np.array([1,1,1])

        print(t)

        for k in range(10000):
            i = random.randint(0,magShape[0]-1)
            j = random.randint(0,magShape[1]-1)
            singleMag = mag[i][j]
            singleAng = ang[i][j]

            if singleMag > THRESHOLD:
                t_train.append([i,j])
                mag_train.append(singleMag)
                ang_train.append(singleAng)

        t_min = 0
        t_max = mags[0].shape[0]
#        t_test = np.linspace(t_min, t_max, 300)

#        y_robust = generate_data(t_test, *res_robust.x)

        mag_robust = least_squares(linearFunc2d, x0, loss='cauchy', \
            f_scale=1, args=(np.array(t_train).transpose(), np.array(mag_train)))
        ang_robust = least_squares(linearFunc2d, x0, loss='cauchy', \
            f_scale=1, args=(np.array(t_train).transpose(), np.array(ang_train)))
#        print(res_robust)
#        y_robust = generateData(linearResultFunc, res_robust.x, t_test)

        magHammer = mag_robust.x
        angHammer = ang_robust.x


        if t == 0:
            prevFrame = frame

        else:
            diffFrame = takeSophisticatedDifferenceFrame(frame, prevFrame, \
                magHammer, angHammer, 3)

            viewFrame(diffFrame, adaptiveScaling=True, differenceImage=True,\
                magnification=10, filename="particle_frames/frame_" + \
                padIntegerWithZeros(t-1, 3) + ".png")

            viewFrame(diffFrame, adaptiveScaling=True, differenceImage=True,\
                magnification=10, filename="particle_frames/frame_")


            prevFrame = frame



#        print(magHammer, angHammer)

#        for (t, y) in zip(t_train, y_train):
 #           p.plot(t, y, "bo")
 #       p.plot(t_test, y_robust, "r-")
 #       p.show()

#            print(k)



#            if singleMag > THRESHOLD:
#                p.plot(j, mag[i][j], "bo")



#                    print(singleMag, ang[i][j])

#        p.show()
