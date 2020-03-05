from PIL import Image
from video_magnifier import viewFrame
import numpy as np
from phase_retrieval import hanning2D
from image_distortion_simulator import imageify, doFuncToEachChannel, \
	doFuncToEachChannelTwoInputs
import sys
from blind_deconvolution import getDerivMagImage, convolve2DToeplitz, convolve2DToeplitzFull
from video_processor import batchAndDifferentiate
import pickle

WHITEBOARD_SIMPLE = False
WHITEBOARD_ALLEN = False
PARTICLE_VID = True

def horizontalHighpassMask(imSpatialDimensions, bandSize):
	returnArray = []

	for i in range(imSpatialDimensions[0]):
		if i <= bandSize or imSpatialDimensions[0] - i <= bandSize:
			returnArray.append([0]*imSpatialDimensions[1])
		else:
			returnArray.append([1]*imSpatialDimensions[1])

	return np.array(returnArray)

def verticalHighpassMask(imSpatialDimensions, bandSize):
	returnArray = []

	for i in range(imSpatialDimensions[1]):
		if i <= bandSize or imSpatialDimensions[1] - i <= bandSize:
			returnArray.append([0]*imSpatialDimensions[0])
		else:
			returnArray.append([1]*imSpatialDimensions[0])

	return np.transpose(np.array(returnArray))

def highpassMask2D(imSpatialDimensions, bandSize):
	return np.multiply(horizontalHighpassMask(imSpatialDimensions, bandSize), \
		verticalHighpassMask(imSpatialDimensions, bandSize))

def padPairOfImagesByIJ(im1, im2, i, j):
	paddedIm1 = doFuncToEachChannel(lambda x: np.pad(x, ((i, 0), (j,0))), im1)
	paddedIm2 = doFuncToEachChannel(lambda x: np.pad(x, ((0,i), (0,j))), im2)

	return paddedIm1, paddedIm2

def getArrayOfDirections(im, patchRadius):
	imShape = im.shape

	for i in range(patchRadius, imShape[0]-patchRadius, 2*patchRadius+1):
		for j in range(patchRadius, imShape[1]-patchRadius, 2*patchRadius+1):
			patch = im[i-patchRadius:i+patchRadius, j-patchRadius:j+patchRadius]

			print(i,j)

			viewFrame(patch)


if WHITEBOARD_SIMPLE:

	im = batchAndDifferentiate(np.array(Image.open("/Users/adamyedidia/whiteboard.png")).astype(float), [(30, False), (30, False), (1, False)])

	viewFrame(getDerivMagImage(im), differenceImage=True, adaptiveScaling=True, magnification=1)

	#viewFrame(im, meanSubtraction=True, differenceImage=True, adaptiveScaling=True, magnification=2)

	#viewFrame(im)
	imSpatialDimensions = im.shape[:-1]
	hanningIm = np.multiply(imageify(hanning2D(imSpatialDimensions))/255, im)

	#viewFrame(hanningIm, adaptiveScaling=False)

	fourierIm = np.fft.fft2(hanningIm)


	for i in range(1, 20):

		print(i)
		mask = highpassMask2D(imSpatialDimensions, i)
		#viewFrame(imageify(mask), adaptiveScaling=True)

		maskedFourierIm = np.multiply(imageify(mask)/255, fourierIm)

		#viewFrame(np.real(maskedFourierIm), adaptiveScaling=True)

		resultIm = np.fft.ifft2(maskedFourierIm)

		viewFrame(resultIm, adaptiveScaling=True, differenceImage=True, \
			filename="whiteboard_frames/frame_" + str(i) + ".png")

if WHITEBOARD_ALLEN:
	im1 = batchAndDifferentiate(np.array(Image.open("/Users/adamyedidia/walls/src/IMG_0839.png")).astype(float), [(10, False), (10, False), (1, False)])
	im2 = batchAndDifferentiate(np.array(Image.open("/Users/adamyedidia/walls/src/IMG_0840.png")).astype(float), [(10, False), (10, False), (1, False)])

#	viewFrame(im1, meanSubtraction=True, differenceImage=True, adaptiveScaling=True, magnification=2)
#	viewFrame(im2, meanSubtraction=True, differenceImage=True, adaptiveScaling=True, magnification=2)

#	im1Padded, im2Padded = padPairOfImagesByIJ(im1, im2, 7,33)
	im1Padded, im2Padded = padPairOfImagesByIJ(im1, im2, 0,0)

	viewFrame(im1Padded - im2Padded, differenceImage=True, adaptiveScaling=True, magnification=30)

#	viewFrame(doFuncToEachChannelTwoInputs(convolve2DToeplitzFull, im1, im2), adaptiveScaling=True)

#	viewFrame(im1)

if PARTICLE_VID:
	vid = pickle.load(open("particle_vid.p", "rb"))

#	im1 = vid[0]
#	im2 = vid[1]
#	im2Padded, im1Padded = padPairOfImagesByIJ(im2, im1, 1,1)

	im = vid[0]
	viewFrame(im)

	getArrayOfDirections(im, 10)

#	viewFrame(im1Padded - im2Padded, differenceImage=True, adaptiveScaling=True, magnification=30)

