from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
import numpy as np
from math import sqrt
from image_distortion_simulator import imageify
import pickle
from PIL import Image
from video_processor import batchArrayAlongAxis, batchAndDifferentiate, convertArrayToVideo

SQUARE_CIRCLE = False
CARLSEN_CIRCLE = False
SQUARE_CIRCLE_CENTER_LOCS = False
BLANK_CIRCLE = True

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

	return returnArray

def getMovingCircleCenter(startingPoint, movementVector, time, arrShape):
	currentPoint = startingPoint + time*movementVector

	return np.array([currentPoint[0]%arrShape[0], currentPoint[1]%arrShape[1]])

def movingCircleNoWrap(radius, startingPoint, movementVector, time, arrShape):
	returnArray = []

	currentPoint = startingPoint + time*movementVector

	for i in range(arrShape[0]):
		returnArray.append([])
		for j in range(arrShape[1]):
#			print sqrt((i - currentPoint[0])**2 + \
#				(j - currentPoint[1])**2), radius

			if sqrt((i - (currentPoint[0]%arrShape[0]))**2 + \
				(j - (currentPoint[1]%arrShape[1]))**2) < radius:

				returnArray[-1].append(1)

			else:
				returnArray[-1].append(0)

	returnArray = np.array(returnArray)

	return returnArray

def mux(a, b, m):
	if m > 0:
		return a
	else:
		return b

def arrMux(arr1, arr2, muxArr):
	return np.vectorize(mux)(arr1, arr2, muxArr)

def makeSquareCircleFrame(numSquaresX, numSquaresY, squareX, squareY, radius, \
	startingPoint, movementVector, time, frameShape):
	
	sqBck = squareBackground(numSquaresX, numSquaresY, squareX, squareY)
	mvCirc = movingCircleNoWrap(radius, startingPoint, movementVector, time, frameShape)

	if time == 50:
		viewFrame(imageify(mvCirc))

	returnArr = arrMux(np.ones(frameShape)*128, sqBck, imageify(mvCirc))

	return returnArr

def makeSquareCircleVideo(numSquaresX, numSquaresY, squareX, squareY, radius, \
	startingPoint, movementVector, numFrames, frameShape):
	
	returnVid = []

	for time in range(numFrames):
		frame = makeSquareCircleFrame(numSquaresX, numSquaresY, squareX, squareY, radius, \
			startingPoint, movementVector, time, frameShape)

		returnVid.append(frame)

		if time == 100:
			viewFrame(frame)

	return np.array(returnVid)

def makeCircleBackgroundFrame(backgroundImage, radius, \
	startingPoint, movementVector, time, frameShape):

	mvCirc = movingCircleNoWrap(radius, startingPoint, movementVector, time, frameShape)

	returnArr = arrMux(np.ones(frameShape)*128, backgroundImage, imageify(mvCirc))	

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

	pickle.dump(vid, open("circle_square_nowrap_vid.p", "w"))

if SQUARE_CIRCLE_CENTER_LOCS:
	numSquaresX = 3
	numSquaresY = 3
	squareX = 11
	squareY = 11
	radius = 5
	startingPoint = np.array([0,0])
	movementVector = np.array([1.1,0.8])
	numFrames = 200
	frameShape = np.array([33, 33, 3])

	listOfCenterLocs = []	

	for time in range(numFrames):
		listOfCenterLocs.append(getMovingCircleCenter(startingPoint, movementVector, \
			time, frameShape))

	pickle.dump(listOfCenterLocs, open("circle_square_center_locs.p", "w"))

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

	pickle.dump(vid, open("circle_carlsen_nowrap_vid.p", "w"))

if BLANK_CIRCLE:
	backgroundImage = np.zeros((10, 10, 3))

	frameShape = backgroundImage.shape

	radius = 3
	startingPoint = np.array([0,0])
	movementVector = np.array([1.1,0.8])
	numFrames = 200

	vid = makeCircleBackgroundVideo(backgroundImage, radius, \
		startingPoint, movementVector, numFrames, frameShape)

	pickle.dump(vid, open("circle_nowrap_vid_10_10.p", "w"))




#	viewFrame(backgroundImage)