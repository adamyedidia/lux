from __future__ import division
import numpy as np
import pickle 
from blind_deconvolution import vectorizedDot, getPseudoInverse, getGaussianKernelVariableLocation, \
	makeImpulseFrame, vectorizedDotToVector
from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
from image_distortion_simulator import imageify, imageifyComplex, \
	doFuncToEachChannelVec, doFuncToEachChannel, circleSpeck, \
    getAttenuationMatrixOneOverF, getAttenuationMatrix, swapChannels, doFuncToEachChannelSeparated, \
    doSeparateFuncToEachChannel, doSeparateFuncToEachChannelSeparated, doFuncToEachChannelSeparatedTwoInputs, \
    doFuncToEachChannelTwoInputs, separate, doSeparateFuncToEachChannelActuallySeparated
import random
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as p
from matplotlib.colors import LinearSegmentedColormap
from math import floor, log
import sys
from custom_plot import createCMapDictHelix


MAKE_GAUSS_VID = False
MAKE_IMPULSE_VID = False
SIM = False
JUMBLE_VIDEO = False
TSNE_JUMBLED = True
MAKE_OBS = False
BUILD_AHAT_FROM_DIFF = False
JUMBLED_RECOVERY = False
TSNE_JUMBLED_APPROX = False

def makeSparseTransferMatrix(shape, fractionOfZeros):
	return np.random.binomial(1, fractionOfZeros, shape)

def permuteArray(arr, permutation):
	returnArray = []

	for i in permutation:
		returnArray.append(arr[i])

	return np.array(returnArray)

def jumbleVideo(vid):
	frameDims = vid[0].shape[:-1]

	assert len(frameDims) == 2

	permutation = range(frameDims[0]*frameDims[1])

	random.shuffle(permutation)

	jumbledVid = []

#	print permutation

	for frame in vid:
		jumbledFrame = doFuncToEachChannel(lambda x: np.reshape(permuteArray(x.flatten(), \
			permutation), frameDims), frame)

		jumbledVid.append(jumbledFrame)

	return np.array(jumbledVid)

def makeListOfXYInRange(arr, minX, maxX, minY, maxY, newShape):
	returnArray = []

	for point in arr:
		x = point[0]
		y = point[1]

		newX = (x - minX)/(maxX - minX)
		newY = (y - minY)/(maxY - minY)

		returnArray.append([newX*newShape[0], newY*newShape[1]])

	return np.array(returnArray)

def downsizePixelMapping(pixelMapping, newShape):
	transPixelMapping = np.transpose(pixelMapping)

	xs = transPixelMapping[0]
	ys = transPixelMapping[1]

	minX = min(xs)
	maxX = max(xs)

	minY = min(ys)
	maxY = max(ys)

	return makeListOfXYInRange(pixelMapping, minX, maxX, minY, maxY, newShape)

def getGaussianKernels(pixelMapping, gaussSD, recoveredFrameShape):
	listOfGaussianKernels = []

	compensationFactor = np.zeros(recoveredFrameShape)

	for point in pixelMapping:
		gaussianKernel = getGaussianKernelVariableLocation(recoveredFrameShape, point, gaussSD)

		listOfGaussianKernels.append(gaussianKernel)

		compensationFactor += gaussianKernel

	return listOfGaussianKernels, compensationFactor

def recoverFrameFromEmbeddingFlat(jumbledFlatFrame, gaussianKernels, compensationFactor, \
	recoveredFrameShape):

	colorKernels = []

	flatFrame = np.transpose(jumbledFlatFrame)

	for colorIndex in range(3):
		kernelSum = np.zeros(recoveredFrameShape)

#		print ("r", "g", "b")[colorIndex]

		for i, pixel in enumerate(flatFrame):

#			print pixel

			pixelVal = pixel[colorIndex]

#			print i, pixelMapping[i], len(pixelMapping)

			kernel = pixelVal*gaussianKernels[i]

			kernelSum += kernel

#		viewFrame(imageify(overallKernel), adaptiveScaling=True)

		compensatedKernel = np.divide(kernelSum, compensationFactor)

		colorKernels.append(compensatedKernel)

	resultFrame = np.swapaxes(np.swapaxes(np.array(colorKernels), 1, 2), 0, 2)

	return resultFrame

def recoverFrameFromEmbedding(jumbledFrame, gaussianKernels, compensationFactor, \
	recoveredFrameShape):

	frameDims = jumbledFrame.shape[:-1]

	flattenedFrame = np.reshape(np.swapaxes(jumbledFrame, 0, 1), (frameDims[0]*frameDims[1], 3)) 

	colorKernels = []

	for colorIndex in range(3):
		kernelSum = np.zeros(recoveredFrameShape)

#		print ("r", "g", "b")[colorIndex]

		for i, pixel in enumerate(flattenedFrame):

#			print pixel

			pixelVal = pixel[colorIndex]

#			print i, pixelMapping[i], len(pixelMapping)

			kernel = pixelVal*gaussianKernels[i]

			kernelSum += kernel

#		viewFrame(imageify(overallKernel), adaptiveScaling=True)

		compensatedKernel = np.divide(kernelSum, compensationFactor)

		colorKernels.append(compensatedKernel)

	resultFrame = np.swapaxes(np.swapaxes(np.array(colorKernels), 1, 2), 0, 2)

	return resultFrame

def recoverVideoFromEmbedding(jumbledVid, pixelMapping, gaussSD, recoveredFrameShape):
	print "computing Gaussian kernels..."

	gaussianKernels, compensationFactor = getGaussianKernels(pixelMapping, gaussSD, recoveredFrameShape)

	returnVid = []

	print "recovering video..."

	for i, jumbledFrame in enumerate(jumbledVid):
		print i

		recoveredFrame = recoverFrameFromEmbedding(jumbledFrame, gaussianKernels, \
			compensationFactor, recoveredFrameShape)

		returnVid.append(recoveredFrame)

#		viewFrame(recoveredFrame, adaptiveScaling=True, magnification=1)

	return returnVid

def recoverVideoFromEmbeddingFlat(jumbledVid, pixelMapping, gaussSD, recoveredFrameShape):
	print "computing Gaussian kernels..."

	gaussianKernels, compensationFactor = getGaussianKernels(pixelMapping, gaussSD, recoveredFrameShape)

	returnVid = []

	print "recovering video..."

	for i, jumbledFrame in enumerate(jumbledVid):
		print i

		recoveredFrame = recoverFrameFromEmbeddingFlat(jumbledFrame, gaussianKernels, \
			compensationFactor, recoveredFrameShape)

		returnVid.append(recoveredFrame)

#		viewFrame(recoveredFrame, adaptiveScaling=True, magnification=1)

	return returnVid

def plotEmbedding(embedding, frameDims):
	for i, point in enumerate(embedding):
#		print "point", point
		p.plot(point[0], point[1], marker="o", color=(floor(i/frameDims[0])/frameDims[1], (i%frameDims[0])/frameDims[0], 0))

	p.show()

def plotEmbeddingApprox(embedding):
#	diffVid = pickle.load(open("steven_batched_coarse_diff.p", "r"))

	cdict5 = createCMapDictHelix(10)

	helix = LinearSegmentedColormap("helix", cdict5)

	p.register_cmap(cmap=helix)

	for i, point in enumerate(embedding):
#		viewFrame(diffVid[i], adaptiveScaling=True)

#		ax = p.gca()
#		ax.set_xlim([0,100])
#		ax.set_ylim([0,100])
#		print "point", point
#		p.plot(point[0], point[1], marker="o", color=helix(i/len(embedding)))
		p.plot(point[0], point[1], marker="o", color=helix(i/len(embedding)))

#		p.show()

	p.show()


if MAKE_GAUSS_VID:
	vid = pickle.load(open("steven_batched_coarse.p", "r"))
	frameDims = vid[0].shape[:-1]

	returnVid = []
	SD = 0.1

	for i in range(frameDims[0]):
		for j in range(frameDims[1]):
#			viewFrame(imageify(getGaussianKernelVariableLocation(frameDims, np.array([i,j]), SD)))

			returnVid.append(imageify(getGaussianKernelVariableLocation(frameDims, np.array([i,j]), SD)))

	pickle.dump(np.array(returnVid), open("moving_gauss_01.p", "w"))

if MAKE_IMPULSE_VID:
	vid = pickle.load(open("steven_batched_coarse.p", "r"))
	frameDims = vid[0].shape[:-1]

	returnVid = []

	for i in range(frameDims[0]):
		for j in range(frameDims[1]):
#			viewFrame(imageify(getGaussianKernelVariableLocation(frameDims, np.array([i,j]), SD)))

			returnVid.append(imageify(makeImpulseFrame(frameDims, (i, j))))

	pickle.dump(np.array(returnVid), open("moving_impulse.p", "w"))

if SIM:
	vid = pickle.load(open("steven_batched_coarse.p", "r"))
	diffVid = pickle.load(open("steven_batched_coarse_diff.p", "r"))
#	diffVid = pickle.load(open("moving_gauss_5.p", "r"))
#	diffVid = pickle.load(open("moving_impulse.p", "r"))

#	print diffVid.shape

	frameDims = diffVid[0].shape[:-1]
	inputVectorSize = frameDims[0]*frameDims[1]

	transferMatShape = (inputVectorSize, inputVectorSize)

	transferMat = makeSparseTransferMatrix(transferMatShape, 0.01)

#	viewFrame(imageify(transferMat), adaptiveScaling=True)

#	print frameDims

#	correctInversionMat = getPseudoInverse(transferMat, 1e10)

	proxyMats = [[], [], []]

	for diffFrame in diffVid:
#		viewFrame(diffFrame)
#		print diffFrame.shape

		obsSeparated = doFuncToEachChannelSeparated(lambda x: vectorizedDot(transferMat, x, frameDims), \
			diffFrame)

		obs = np.swapaxes(np.swapaxes(obsSeparated, 1, 2), 0, 2)


#		print obs.shape

#		viewFrame(obs, adaptiveScaling=True)

		proxyMats[0].append(obsSeparated[0].flatten())
		proxyMats[1].append(obsSeparated[1].flatten())
		proxyMats[2].append(obsSeparated[2].flatten())

	proxyMats = [np.transpose(np.array(proxyMat)) for proxyMat in proxyMats]

	viewFrame(imageify(proxyMats[0]), adaptiveScaling=True)

	print proxyMats[0].shape

	recoveredInversionMats = [getPseudoInverse(proxyMat, 1e-8) for proxyMat in proxyMats]

#	recoveredInversionMats = [correctInversionMat]*3

	for frame in vid:
		obs = doFuncToEachChannel(lambda x: vectorizedDot(transferMat, x, frameDims), \
			frame)
#		obs = frame

		recovery = doSeparateFuncToEachChannelSeparated([lambda x: \
			vectorizedDot(recoveredInversionMats[i], x, frameDims) for i in range(3)], obs)

		viewFrame(recovery, adaptiveScaling=True)

if JUMBLE_VIDEO:
	vid = pickle.load(open("steven_batched_coarse.p", "r"))

	jumbledVid = jumbleVideo(vid)

	pickle.dump(jumbledVid, open("steven_batched_coarse_jumbled.p", "w"))

if MAKE_OBS:
	vid = pickle.load(open("steven_batched_coarse.p", "r"))

	frameDims = vid[0].shape[:-1]
	inputVectorSize = frameDims[0]*frameDims[1]

	transferMatShape = (inputVectorSize, inputVectorSize)

	transferMat = makeSparseTransferMatrix(transferMatShape, 0.1)

	viewFrame(imageify(transferMat), adaptiveScaling=True)

	obsVid = []

	for frame in vid:

		obs = doFuncToEachChannel(lambda x: vectorizedDot(transferMat, x, frameDims), \
			frame)

		obsVid.append(obs)

	pickle.dump(np.array(obsVid), open("steven_batched_coarse_obs_noiseless.p", "w"))

if TSNE_JUMBLED:
#	jumbledVid = pickle.load(open("steven_batched_coarse_jumbled.p", "r"))
#	jumbledVid = pickle.load(open("steven_batched_coarse.p", "r"))
#	jumbledVid = pickle.load(open("circle_batched.p", "r"))
#	jumbledVid = pickle.load(open("circle_square_vid.p", "r"))
#	jumbledVid = np.array(pickle.load(open("circle_square_vid_meansub.p", "r")))
#	jumbledVid = pickle.load(open("circle_square_vid_diff.p", "r"))
#	jumbledVid = pickle.load(open("circle_carlsen_vid.p", "r"))
	jumbledVid = np.array(pickle.load(open("circle_carlsen_vid_meansub.p", "r")))

#	print jumbledVid.shape

	vidShape = jumbledVid.shape

	recoveryShape = (100, 100)

	numFrames = vidShape[0]
	frameDims = vidShape[1:3]

	flatVid = np.reshape(jumbledVid, (numFrames, frameDims[0]*frameDims[1], 3))

	colorStackedVid = np.reshape(np.swapaxes(jumbledVid, 1, 3), (numFrames*3, frameDims[0]*frameDims[1]))

#	print colorStackedVid.shape

#	viewFlatFrame(imageify(colorStackedVid[3])/255)
#	viewFlatFrame(flatVid[3])

	X = np.transpose(colorStackedVid)


#	X = np.random.random((20,20))
#	print X
#	X_embedded = TSNE(n_components=2, verbose=6).fit_transform(X)
#	print X_embedded.shape
#	print X_embedded

#	X, _ = load_digits(return_X_y=True)

#	X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
	print X.shape

	embedding = Isomap(n_components=2, n_neighbors=10)
#	embedding = MDS(n_components=2, metric=False, verbose=True)
#	embedding = PCA(n_components=2)
#	embedding = LocallyLinearEmbedding(n_components=2, reg=1e2, n_neighbors=20)
#	embedding = TSNE(n_components=2, verbose=2, init="pca", learning_rate=1000, early_exaggeration=1, \
#		perplexity=50)

	correctEmbedding = []

	for j in range(frameDims[1]):
		for i in range(frameDims[0]):
			correctEmbedding.append((i, j))

	X_transformed = embedding.fit_transform(X)
#	print X_transformed.shape	
#	print X_transformed
	X_downsized = downsizePixelMapping(X_transformed, recoveryShape)

	if True:

		plotEmbedding(X_downsized, frameDims)

#		for point in X_transformed:
#			print "point", point
#			p.plot(point[0], point[1], "ro")
#		p.show()

#		for point in X_downsized:
#			print "point", point
#			p.plot(point[0], point[1], "bo")
#		p.show()

	recoveredVid = recoverVideoFromEmbedding(jumbledVid[3:], X_downsized, 2, recoveryShape)

	pickle.dump(recoveredVid, open("recovered_vid.p", "w"))

#		viewFrame(diffFrame)
#		viewFrame(obs, adaptiveScaling=True)



#		recovery = doFuncToEachChannel(lambda x: vectorizedDot(correctInversionMat, x, frameDims), \
#			obs)

#		viewFrame(recovery, adaptiveScaling=True)

if BUILD_AHAT_FROM_DIFF:
#	vid = pickle.load(open("steven_batched_coarse.p", "r"))
	diffObs = pickle.load(open("steven_batched_coarse_obs_noiseless_diff.p", "r"))
#	diffVid = pickle.load(open("moving_gauss_5.p", "r"))
#	diffVid = pickle.load(open("moving_impulse.p", "r"))

	diffVid = pickle.load(open("steven_batched_coarse_diff.p", "r"))

#	print diffVid.shape

	frameDims = diffObs[0].shape[:-1]
	inputVectorSize = frameDims[0]*frameDims[1]

	transferMatShape = (inputVectorSize, inputVectorSize)

	transferMat = makeSparseTransferMatrix(transferMatShape, 0.01)

#	viewFrame(imageify(transferMat), adaptiveScaling=True)

#	print frameDims

#	correctInversionMat = getPseudoInverse(transferMat, 1e10)

	proxyMats = [[], [], []]

	for i, diffObsFrame in enumerate(diffObs):
		obsNorm = np.sum(np.multiply(diffObsFrame, diffObsFrame))

		diffFrame = diffVid[i]

#		print i, log(obsNorm+1e-5)
#		viewFrame(diffFrame, adaptiveScaling=True)
#		viewFrame(diffObsFrame, adaptiveScaling=True)
#		print diffFrame.shape

		obsSeparated = separate(diffObsFrame)

#		viewFrame(imageify(obsSeparated[0]), adaptiveScaling=True)

		if log(obsNorm+1e-5) < 18 and log(obsNorm+1e-5) > 10:

			print True
#		if obsNorm > 0:

#		print obs.shape

#		viewFrame(obs, adaptiveScaling=True)

			proxyMats[0].append(obsSeparated[0].flatten())
			proxyMats[1].append(obsSeparated[1].flatten())
			proxyMats[2].append(obsSeparated[2].flatten())

#			print len(obsSeparated[0].flatten())

#			viewFlatFrame(imageify(obsSeparated[0].flatten()/obsNorm), magnification=2550)

		else:
			print False

	proxyMats = [np.transpose(np.array(proxyMat)) for proxyMat in proxyMats]

	viewFrame(imageify(proxyMats[0]), adaptiveScaling=True)

	print proxyMats[0].shape

	recoveredInversionMats = [getPseudoInverse(proxyMat, 1e-7) for proxyMat in proxyMats]

	pickle.dump(recoveredInversionMats, open("steven_approx_inv.p", "w")) 

if JUMBLED_RECOVERY:
	obsVid = pickle.load(open("steven_batched_coarse_obs_noiseless.p", "r"))

	recoveredInversionMats = pickle.load(open("steven_approx_inv.p", "r"))

	recoveredVecsSeparated = []
	recoveredVecsGrouped = []

	for obsFrame in obsVid:
		recoveredVecSeparated = doSeparateFuncToEachChannelActuallySeparated([lambda x: \
			vectorizedDotToVector(recoveredInversionMats[i], x) for i in range(3)], 
			obsFrame)

		recoveredVecsSeparated.append(recoveredVecSeparated[0])
		recoveredVecsSeparated.append(recoveredVecSeparated[1])
		recoveredVecsSeparated.append(recoveredVecSeparated[2])

		recoveredVecsGrouped.append(recoveredVecSeparated)

	pickle.dump(np.array(recoveredVecsSeparated), \
		open("steven_batched_coarse_obs_noiseless_jumbled_recovery_separated.p", "w"))

	pickle.dump(np.array(recoveredVecsGrouped), \
		open("steven_batched_coarse_obs_noiseless_jumbled_recovery_grouped.p", "w"))

if TSNE_JUMBLED_APPROX:
	jumbledRecoverySeparated = \
		pickle.load(open("steven_batched_coarse_obs_noiseless_jumbled_recovery_separated.p", "r"))

	jumbledRecoveryGrouped = \
		pickle.load(open("steven_batched_coarse_obs_noiseless_jumbled_recovery_grouped.p", "r"))

	recoveryShape = (100, 100)

	X = np.transpose(jumbledRecoverySeparated)

	print X.shape

#	embedding = Isomap(n_components=2, n_neighbors=3)
#	embedding = MDS(n_components=2, metric=False)
	embedding = PCA(n_components=2)
#	embedding = LocallyLinearEmbedding(n_components=2, reg=0.001, n_neighbors=4)
#	embedding = TSNE(n_components=2, verbose=2, init="pca", learning_rate=1000, early_exaggeration=1, \
#		perplexity=50)

	X_transformed = embedding.fit_transform(X)
	X_downsized = downsizePixelMapping(X_transformed, recoveryShape)

	if True:

		plotEmbeddingApprox(X_downsized)

#		for point in X_transformed:
#			print "point", point
#			p.plot(point[0], point[1], "ro")
#		p.show()

#		for point in X_downsized:
#			print "point", point
#			p.plot(point[0], point[1], "bo")
#		p.show()

	recoveredVid = recoverVideoFromEmbeddingFlat(jumbledRecoveryGrouped, X_downsized, 2, recoveryShape)

	pickle.dump(recoveredVid, open("recovered_vid.p", "w"))

#		viewFrame(diffFrame)
#		viewFrame(obs, adaptiveScaling=True)



#		recovery = doFuncToEachChannel(lambda x: vectorizedDot(correctInversionMat, x, frameDims), \
#			obs)

#		viewFrame(recovery, adaptiveScaling=True)

