from __future__ import division
from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
from image_distortion_simulator import imageify, imageifyComplex
import numpy as np
from PIL import Image
import pickle

VERTICALLY_PROJECT_VID = False
HORIZONTALLY_PROJECT_VID = False
DOUBLE_PROJECT = False
DOUBLE_DORA = True

def verticallyAverageFrame(frame):
	averageVals = np.sum(frame, 0)/frame.shape[0]

	return np.array([averageVals]*frame.shape[0])

def horizontallyAverageFrame(frame):
	averageVals = np.sum(frame, 1)/frame.shape[1]

	return np.swapaxes(np.array([averageVals]*frame.shape[1]), 0, 1)

def naiveDoubleProjection(frame):
	vert = verticallyAverageFrame(frame)
	horiz = horizontallyAverageFrame(frame)

	return (vert + horiz) / 2

if VERTICALLY_PROJECT_VID:
	arrName = "steven_batched"

	vid = pickle.load(open(arrName + ".p", "r"))

	newVid = []

	for frame in vid:
		newVid.append(verticallyAverageFrame(frame))

	pickle.dump(newVid, open(arrName + "_vert.p", "w"))

if HORIZONTALLY_PROJECT_VID:
	arrName = "steven_batched"

	vid = pickle.load(open(arrName + ".p", "r"))

	newVid = []

	for i, frame in enumerate(vid):
		if i == 200:
			viewFrame(horizontallyAverageFrame(frame))

		newVid.append(horizontallyAverageFrame(frame))

	pickle.dump(newVid, open(arrName + "_horiz.p", "w"))	

if DOUBLE_PROJECT:
	arrName = "steven_batched"

	vid = pickle.load(open(arrName + ".p", "r"))

	newVid = []

	for i, frame in enumerate(vid):
		if i == 200:
			viewFrame(naiveDoubleProjection(frame))

		newVid.append(naiveDoubleProjection(frame))

	pickle.dump(newVid, open(arrName + "_double.p", "w"))	

if DOUBLE_DORA:
    im = pickle.load(open("dora_very_downsampled.p", "r"))

    viewFrame(naiveDoubleProjection(im), adaptiveScaling=True, magnification=1)