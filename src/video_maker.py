from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
import numpy as np
from math import sqrt
from image_distortion_simulator import imageify
import pickle
from PIL import Image
from video_processor import batchArrayAlongAxis, batchAndDifferentiate, convertArrayToVideo

SQUARE_CIRCLE = False
CARLSEN_CIRCLE = True

def numberToColor(x):
	r = int(x % 8) >> 2
	g = int(x % 4) >> 1
	b = int(x % 2)

	return np.array([r*255, g*255, b*255])

def squareBackground(numSquaresX, numSquaresY, squareX, squareY):
	nugget = []

	for i in range(numSquaresX):
		nugget.append([])
		for j in range(numSquaresY):
			nugget[-1].append(numberToColor(i*numSquaresY + j))

	nugget = np.repeat(np.array(nugget), squareX, axis=0)
	nugget = np.repeat(nugget, squareY, axis=1)

#	viewFrame(nugget)

	return nugget

def modularAbsDifference(x, y, mod):
	return min((x - y) % mod, (y - x) % mod)

def movingCircle(radius, startingPoint, movementVector, time, arrShape):
	returnArray = []

	currentPoint = startingPoint + time*movementVector

	for i in range(arrShape[0]):
		returnArray.append([])
		for j in range(arrShape[1]):
			if sqrt(modularAbsDifference(i, currentPoint[0], arrShape[0])**2 + \
				modularAbsDifference(j, currentPoint[1], arrShape[1])**2) < radius:

				returnArray[-1].append(1)

			else:
				returnArray[-1].append(0)

	returnArray = np.array(returnArray)


#	viewFrame(imageify(returnArray))

	return imageify(returnArray)

def mux(a, b, m):
	if m:
		return a
	else:
		return b

def arrMux(arr1, arr2, muxArr):
	return np.vectorize(mux)(arr1, arr2, muxArr)

def makeSquareCircleFrame(numSquaresX, numSquaresY, squareX, squareY, radius, \
	startingPoint, movementVector, time, frameShape):
	
	sqBck = squareBackground(numSquaresX, numSquaresY, squareX, squareY)
	mvCirc = movingCircle(radius, startingPoint, movementVector, time, frameShape)

	returnArr = arrMux(np.ones(frameShape)*128, sqBck, mvCirc)	

	return returnArr

def makeSquareCircleVideo(numSquaresX, numSquaresY, squareX, squareY, radius, \
	startingPoint, movementVector, numFrames, frameShape):
	
	returnVid = []

	for time in range(numFrames):
		frame = makeSquareCircleFrame(numSquaresX, numSquaresY, squareX, squareY, radius, \
			startingPoint, movementVector, time, frameShape)

		returnVid.append(frame)

	return np.array(returnVid)

def makeCircleBackgroundFrame(backgroundImage, radius, \
	startingPoint, movementVector, time, frameShape):

	mvCirc = movingCircle(radius, startingPoint, movementVector, time, frameShape)

	returnArr = arrMux(np.ones(frameShape)*128, backgroundImage, mvCirc)	

	return returnArr

def makeCircleBackgroundVideo(backgroundImage, radius, \
	startingPoint, movementVector, numFrames, frameShape):

	returnVid = []

	for time in range(numFrames):
		frame = makeCircleBackgroundFrame(backgroundImage, radius, \
			startingPoint, movementVector, time, frameShape)

		returnVid.append(frame)

	return np.array(returnVid)



if SQUARE_CIRCLE:
	numSquaresX = 3
	numSquaresY = 3
	squareX = 11
	squareY = 11
	radius = 5
	startingPoint = np.array([0,0])
	movementVector = np.array([1.1,0.8])
	numFrames = 200
	frameShape = np.array([33, 33, 3])

	vid = makeSquareCircleVideo(numSquaresX, numSquaresY, squareX, squareY, radius, \
		startingPoint, movementVector, numFrames, frameShape)

	pickle.dump(vid, open("circle_square_vid.p", "w"))

if CARLSEN_CIRCLE:
	backgroundImage = batchAndDifferentiate(np.array(Image.open("carlsen_caruana.jpg")).astype(float), \
		[(40, False), (40, False), (1, False)])

	frameShape = backgroundImage.shape

	radius = 5
	startingPoint = np.array([0,0])
	movementVector = np.array([1.1,0.8])
	numFrames = 200

	vid = makeCircleBackgroundVideo(backgroundImage, radius, \
		startingPoint, movementVector, numFrames, frameShape)

	pickle.dump(vid, open("circle_carlsen_vid.p", "w"))




#	viewFrame(backgroundImage)