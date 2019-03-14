from __future__ import division
from PIL import Image
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
from best_matrix import padIntegerWithZeros
from scipy.optimize import minimize


MAKE_GAUSS_VID = False
MAKE_IMPULSE_VID = False
SIM = False
JUMBLE_VIDEO = False
TSNE_REMOVE_OUTLIERS = False
TSNE_JUMBLED = False
MAKE_OBS = False
BUILD_AHAT_FROM_DIFF = False
JUMBLED_RECOVERY = False
TSNE_JUMBLED_APPROX = False
RECOVERY_WITH_RIGHT_CENTERS = False
SVDS = False
SVDS_BW = False
TSNE_JUMBLED_SVD = False
MAKE_MOVING_IMPULSE_VID = False
MAKE_GLASS_ROSE_GT_XFER_MAT = False
GT_RECON = False
GET_LINEAR_COMBS = False
GET_BASIS_VEC_FROM_SPARSITY = True

def turnVidToMatrix(vid):
	returnArray = []

	for frame in vid:
		returnArray.append(frame.flatten())

	returnArray = np.array(returnArray)

	print returnArray.shape

	return returnArray

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

def recoverVideoFromEmbedding(jumbledVid, pixelMapping, gaussSD, recoveredFrameShape,
	xBounds=None, yBounds=None):
	print "computing Gaussian kernels..."

	gaussianKernels, compensationFactor = getGaussianKernels(pixelMapping, gaussSD, recoveredFrameShape)

	returnVid = []

	print "recovering video..."

	for i, jumbledFrame in enumerate(jumbledVid):
		print i

		recoveredFrame = recoverFrameFromEmbedding(jumbledFrame, gaussianKernels, \
			compensationFactor, recoveredFrameShape)

		if xBounds == None:
			truncatedFrame = recoveredFrame

		else:
			truncatedFrame = recoveredFrame[xBounds[0]:xBounds[1],yBounds[0]:yBounds[1]]

#		if i > 5:	
#			viewFrame(truncatedFrame, adaptiveScaling=True)

		returnVid.append(truncatedFrame)

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

def removeOutliers(embedding, median, outlierDistance):
	newEmbedding = []

	for i, point in enumerate(embedding):
		print np.linalg.norm(median - point)
		print outlierDistance
		print np.linalg.norm(median - point) < outlierDistance

		if np.linalg.norm(median - point) < outlierDistance:
			newEmbedding.append(point)

	return newEmbedding

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

def rewardFunctionMaker(svs, svReward1, svReward2, sparsityPenalty):
	def rewardFunction(svVec):
		dotResult = np.dot(svs, svVec)


		penaltyVal = -svReward1*np.sqrt(np.abs(svVec[0])) - \
			svReward2*np.sqrt(np.sum(np.abs(svVec))) + \
			sparsityPenalty*np.sum(np.abs(dotResult))

		print penaltyVal

		return penaltyVal

	return rewardFunction


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
	vid = pickle.load(open("circle_square_nowrap_vid.p", "r"))
	diffVid = pickle.load(open("circle_square_nowrap_vid_diff.p", "r"))

	frameDims = vid[0].shape[:-1]
	inputVectorSize = frameDims[0]*frameDims[1]

	transferMatShape = (inputVectorSize, inputVectorSize)

	transferMat = makeSparseTransferMatrix(transferMatShape, 0.1)

	viewFrame(imageify(transferMat), adaptiveScaling=True)

	obsVid = []
	diffObsVid = []

	for frame, diffFrame in zip(vid, diffVid):

		obs = doFuncToEachChannel(lambda x: vectorizedDot(transferMat, x, frameDims), \
			frame)
		diffObs = doFuncToEachChannel(lambda x: vectorizedDot(transferMat, x, frameDims), \
			diffFrame)

		obsVid.append(obs)
		diffObsVid.append(diffObs)

	pickle.dump(np.array(obsVid), open("circle_square_nowrap_vid_obs.p", "w"))
	pickle.dump(np.array(obsVid), open("circle_square_nowrap_vid_diff_obs.p", "w"))

if TSNE_REMOVE_OUTLIERS:
	jumbledVid = np.array(pickle.load(open("glass_rose_avgdiv.p", "r")))


	vidShape = jumbledVid.shape

	recoveryShape = (100, 100)

	numFrames = vidShape[0]
	frameDims = vidShape[1:3]

	flatVid = np.reshape(jumbledVid, (numFrames, frameDims[0]*frameDims[1], 3))

	colorStackedVid = np.reshape(np.swapaxes(jumbledVid, 1, 3), (numFrames*3, frameDims[0]*frameDims[1]))	

	X = np.transpose(colorStackedVid)

	embedding = Isomap(n_components=2, n_neighbors=10)

	correctEmbedding = []

	for j in range(frameDims[1]):
		for i in range(frameDims[0]):
			correctEmbedding.append((i, j))

	X_transformed = embedding.fit_transform(X)
#	print X_transformed.shape	
#	print X_transformed
	X_downsized = downsizePixelMapping(X_transformed, recoveryShape)

	xDownsizedXs = [i[0] for i in X_downsized]
	xDownsizedYs = [i[1] for i in X_downsized]

	medianX = np.median(xDownsizedXs)
	medianY = np.median(xDownsizedYs)

	OUTLIER_DISTANCE = 40

	xOutliersRemoved = removeOutliers(X_downsized, np.array([medianX, medianY]), OUTLIER_DISTANCE)

	if True:

		plotEmbedding(X_downsized, frameDims)
		plotEmbedding(xOutliersRemoved, frameDims)




if TSNE_JUMBLED:

	originalVid = pickle.load(open("prafull_ball_ds_meansub_avgdiv.p", "r"))
#	originalVid = pickle.load(open("circle_batched_ds.p", "r"))
#	originalVid = np.array(pickle.load(open("glass_rose.p", "r")))
#	originalVid = pickle.load(open("circle_square_nowrap_vid.p", "r"))
#	originalVid = pickle.load(open("circle_carlsen_nowrap_vid.p", "r"))	
#	originalVid = pickle.load(open("circle_square_vid.p", "r"))
#	originalVid = pickle.load(open("circle_carlsen_vid.p", "r"))
#	jumbledVid = pickle.load(open("steven_batched_coarse_jumbled.p", "r"))
#	jumbledVid = pickle.load(open("steven_batched_coarse.p", "r"))
	jumbledVid = pickle.load(open("prafull_ball_ds_meansub_avgdiv.p", "r"))
#	jumbledVid = pickle.load(open("circle_batched_ds.p", "r"))
#	jumbledVid = pickle.load(open("circle_square_vid.p", "r"))
#	jumbledVid = np.array(pickle.load(open("circle_square_vid_meansub.p", "r")))
#	jumbledVid = pickle.load(open("circle_square_vid_diff.p", "r"))
#	jumbledVid = pickle.load(open("circle_carlsen_vid.p", "r"))
#	jumbledVid = np.array(pickle.load(open("circle_carlsen_vid_meansub.p", "r")))
#	jumbledVid = np.array(pickle.load(open("circle_square_nowrap_vid_meansub_abs_coloravg.p", "r")))
#	jumbledVid = np.array(pickle.load(open("circle_carlsen_vid_meansub_abs_coloravg.p", "r")))
#	jumbledVid = np.array(pickle.load(open("circle_carlsen_nowrap_vid_meansub_abs_coloravg_avgdiv.p", "r")))
#	jumbledVid = np.array(pickle.load(open("glass_rose.p", "r")))
#	jumbledVid = np.array(pickle.load(open("glass_rose_avgdiv.p", "r")))
#	jumbledVid = np.array(pickle.load(open("glass_rose_meansub_avgdiv.p", "r")))

#	print jumbledVid.shape

	originalVid = np.array(originalVid)
	jumbledVid = np.array(jumbledVid)

	vidShape = jumbledVid.shape

	print vidShape

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

	embedding = Isomap(n_components=2, n_neighbors=5)
#	embedding = MDS(n_components=2, metric=False, verbose=True) # careful this one takes forever
#	embedding = PCA(n_components=2)
#	embedding = LocallyLinearEmbedding(n_components=2, reg=1e-3, n_neighbors=5)
#	embedding = TSNE(n_components=2, verbose=2, learning_rate=1000, early_exaggeration=1, \
#		perplexity=300, method='exact')

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

#	recoveredVid = recoverVideoFromEmbedding(originalVid[3:], X_downsized, 2, recoveryShape)
	recoveredVid = recoverVideoFromEmbedding(jumbledVid[3:], X_downsized, 2, recoveryShape)
#		xBounds=[110,170], yBounds=[0,160])
#		xBounds=[0,80], yBounds=[55,85])


	pickle.dump(recoveredVid, open("recovered_vid.p", "w"))

#		viewFrame(diffFrame)
#		viewFrame(obs, adaptiveScaling=True)



#		recovery = doFuncToEachChannel(lambda x: vectorizedDot(correctInversionMat, x, frameDims), \
#			obs)

#		viewFrame(recovery, adaptiveScaling=True)

if BUILD_AHAT_FROM_DIFF:
	vid = pickle.load(open("circle_square_vid.p", "r"))
	diffObs = pickle.load(open("circle_square_nowrap_vid_diff_obs.p", "r"))
#	diffVid = pickle.load(open("moving_gauss_5.p", "r"))
#	diffVid = pickle.load(open("moving_impulse.p", "r"))

#	diffVid = pickle.load(open("steven_batched_coarse_diff.p", "r"))

	diffVid = pickle.load(open("circle_square_nowrap_vid_diff.p", "r"))

#	print diffVid.shape

	frameDims = diffVid[0].shape[:-1]
	inputVectorSize = frameDims[0]*frameDims[1]

	transferMatShape = (inputVectorSize, inputVectorSize)

	transferMat = makeSparseTransferMatrix(transferMatShape, 0.01)


#	viewFrame(imageify(transferMat), adaptiveScaling=True)

#	print frameDims

#	correctInversionMat = getPseudoInverse(transferMat, 1e10)

#	diffObs = [doFuncToEachChannel(lambda x: vectorizedDot(transferMat, x, frameDims),
#			sceneFrame) for sceneFrame in diffVid]

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

#		print log(obsNorm+1e-5)

		if log(obsNorm+1e-5) < 30 and log(obsNorm+1e-5) > 10:

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

	print np.array(proxyMats).shape

	viewFrame(imageify(proxyMats[0]), adaptiveScaling=True)

	print proxyMats[0].shape

	recoveredInversionMats = [getPseudoInverse(proxyMat, 1e-7) for proxyMat in proxyMats]

	pickle.dump(recoveredInversionMats, open("circle_square_approx_inv.p", "w")) 

if JUMBLED_RECOVERY:
	obsVid = pickle.load(open("circle_square_nowrap_vid_obs.p", "r"))

	recoveredInversionMats = pickle.load(open("circle_square_approx_inv.p", "r"))

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
		open("circle_square_nowrap_vid_obs_jumbled_recovery_separated.p", "w"))

	pickle.dump(np.array(recoveredVecsGrouped), \
		open("circle_square_nowrap_vid_obs_jumbled_recovery_grouped.p", "w"))

if TSNE_JUMBLED_APPROX:
#	jumbledRecoverySeparated = \
#		pickle.load(open("circle_square_nowrap_vid_obs_jumbled_recovery_separated.p", "r"))

	jumbledRecoverySeparated = \
		pickle.load(open("circle_square_nowrap_vid_obs_jumbled_recovery_grouped_meansub_abs_coloravg_avgdiv_colorflat.p", "r"))

	jumbledRecoveryGrouped = \
		pickle.load(open("circle_square_nowrap_vid_obs_jumbled_recovery_grouped.p", "r"))

	print np.array(jumbledRecoverySeparated).shape
	print np.array(jumbledRecoveryGrouped).shape

	recoveryShape = (100, 100)

	X = np.transpose(jumbledRecoverySeparated)

	print X.shape

	embedding = Isomap(n_components=2, n_neighbors=5)
#	embedding = MDS(n_components=2, metric=False)
#	embedding = PCA(n_components=2)
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

if RECOVERY_WITH_RIGHT_CENTERS:
#	jumbledRecoverySeparated = \
#		pickle.load(open("circle_square_nowrap_vid_obs_jumbled_recovery_separated.p", "r"))

	jumbledRecoveryGrouped = \
		pickle.load(open("circle_square_nowrap_vid_obs_jumbled_recovery_grouped.p", "r"))

	listOfCenterLocs = pickle.load(open("circle_square_center_locs.p", "r"))

	recoveryShape = (100, 100)

	embedding = Isomap(n_components=2, n_neighbors=5)
#	embedding = MDS(n_components=2, metric=False)
#	embedding = PCA(n_components=2)
#	embedding = LocallyLinearEmbedding(n_components=2, reg=0.001, n_neighbors=4)
#	embedding = TSNE(n_components=2, verbose=2, init="pca", learning_rate=1000, early_exaggeration=1, \
#		perplexity=50)

	centerLocsDownsized = downsizePixelMapping(listOfCenterLocs, recoveryShape)

	if True:
		plotEmbeddingApprox(centerLocsDownsized)

#		for point in X_transformed:
#			print "point", point
#			p.plot(point[0], point[1], "ro")
#		p.show()

#		for point in X_downsized:
#			print "point", point
#			p.plot(point[0], point[1], "bo")
#		p.show()

	recoveredVid = recoverVideoFromEmbeddingFlat(jumbledRecoveryGrouped, centerLocsDownsized, 2, recoveryShape)

	pickle.dump(recoveredVid, open("recovered_vid.p", "w"))

#		viewFrame(diffFrame)
#		viewFrame(obs, adaptiveScaling=True)



#		recovery = doFuncToEachChannel(lambda x: vectorizedDot(correctInversionMat, x, frameDims), \
#			obs)

#		viewFrame(recovery, adaptiveScaling=True)

if SVDS:
	vid = pickle.load(open("steven_batched_coarse.p", "r"))
	frameShape = vid[0].shape
	print frameShape

	mat = turnVidToMatrix(vid)

	u, s, vh = np.linalg.svd(mat)

	print u.shape
	print vh.shape

	for sv in u[:20]:
		p.plot(sv)
		p.show()

	for sv in vh[:20]:
		viewFrame(np.reshape(sv, frameShape), differenceImage=True, adaptiveScaling=True)

if SVDS_BW:
#	vid = pickle.load(open("glass_rose_2_framesplit.p", "r"))
	vid = pickle.load(open("impulse_movie_framesplit.p", "r"))
	frameShape = vid[0].shape
	print frameShape

	mat = turnVidToMatrix(vid)

	u, s, vh = np.linalg.svd(mat)

	print u.shape
	print vh.shape

	for sv in vh:
		viewFrame(imageify(np.reshape(sv, frameShape)), differenceImage=True, \
			adaptiveScaling=True)

	estimatedTransferMatrix = vh[:100]

#	p.matshow(estimatedTransferMatrix)
#	p.show()

	pickle.dump(estimatedTransferMatrix, open("glass_rose_xfer_sv.p", "w"))

if TSNE_JUMBLED_SVD:

	originalVid = pickle.load(open("glass_rose.p", "r"))
#	jumbledVid = pickle.load(open("prafull_ball_ds_meansub_avgdiv.p", "r"))

	transferMat = np.transpose(pickle.load(open("glass_rose_xfer_sv.p", "r")))

	snr = 1e1

	inverseMat = getPseudoInverse(transferMat, snr)

	for i,frame in enumerate(originalVid):
		print i

		obs = doFuncToEachChannel(lambda x: \
			vectorizedDot(inverseMat, x.flatten(), (5, 6)), \
			frame)

		viewFrame(obs, adaptiveScaling=True, \
			magnification=10, filename="pixel_vid/frame_" + padIntegerWithZeros(i, 3) + ".png")


if MAKE_MOVING_IMPULSE_VID:
	squareSideLength = 160
	xLength = int(2560/squareSideLength)
	yLength = int(1600/squareSideLength)

	xRange = range(xLength)
	yRange = range(yLength)
	background = Image.new('RGB', (squareSideLength*xLength, \
		squareSideLength*yLength), color = (0, 0, 0))
	whiteSquare = Image.new('RGB', (squareSideLength, \
		squareSideLength), color = (255, 255, 255))

	for y in range(yLength):
		for x in range(xLength):
			backgroundCopy = background.copy()
			position = (x*squareSideLength, y*squareSideLength)
			backgroundCopy.paste(whiteSquare, position)
			frameNumber = y*xLength + x


			backgroundCopy.save('impulse_movie/frame_' + \
				padIntegerWithZeros(frameNumber, 3) + ".png")

if MAKE_GLASS_ROSE_GT_XFER_MAT:
	vid = pickle.load(open("glass_rose_calibration.p"))

	print len(vid)

	startFrame = 6
	endFrame = len(vid) - 7

#	viewFrame(vid[startFrame])
#	viewFrame(vid[startFrame + 1])
#	viewFrame(vid[endFrame])
#	viewFrame(vid[endFrame - 1])

	transferMatrix = []

	for i in np.linspace(startFrame, endFrame, 160):
		i = int(round(i))
		frame = vid[i]
		frameR = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)[0]

#		viewFrame(imageify(frameR)/255)

		viewFrame(imageify(frameR)/255)

		transferMatrix.append(frameR.flatten())

	pickle.dump(transferMatrix, open("glass_rose_2_xfer_gt.p", "w"))

if GT_RECON:
	vid = pickle.load(open("glass_rose_2.p"))

#	transferMat = np.transpose(np.array(pickle.load(open("glass_rose_2_xfer_gt.p", "r"))))
	transferMat = np.transpose(np.array(pickle.load(open("glass_rose_2_recovered_basis_100.p", "r"))))

	snr = 1e6

	pseudoInverse = getPseudoInverse(transferMat, snr)

	recoveredVid = []

	for i, frame in enumerate(vid):
		print i

		recoveredFrame = doFuncToEachChannel(lambda x: vectorizedDot(pseudoInverse, x, \
			(10, 16)), frame)

		recoveredVid.append(recoveredFrame)

#		if i % 10 == 0:
#			viewFrame(recoveredFrame, adaptiveScaling=True)

	pickle.dump(np.array(recoveredVid), open("glass_rose_2_recovery_with_gt.p", "w"))

if GET_LINEAR_COMBS:
	vid = pickle.load(open("glass_rose_2_framesplit.p", "r"))
	frameShape = vid[0].shape

	transferMatrixGT = np.array(pickle.load(open("glass_rose_2_xfer_gt.p", "r")))
	SVs = np.array(pickle.load(open("glass_rose_xfer_sv.p", "r")))

	snr = 1e-6

	SVinv = getPseudoInverse(SVs, snr)

	result = np.dot(transferMatrixGT, SVinv)

	p.matshow(result)
	p.colorbar()
	p.show()

	recoveredBasis = np.dot(result, SVs)

	print recoveredBasis.shape

#	for i, recoveredBasisFrame in enumerate(recoveredBasis):
#		print "first"
#		viewFrame(imageify(np.reshape(recoveredBasisFrame, frameShape))/255, adaptiveScaling=True)
#		print "second"
#		viewFrame(imageify(np.reshape(transferMatrixGT[i], frameShape)), adaptiveScaling=True)

	pickle.dump(recoveredBasis, open("glass_rose_2_recovered_basis_100.p", "w"))

if GET_BASIS_VEC_FROM_SPARSITY:
	vid = pickle.load(open("glass_rose_2_framesplit.p", "r"))
	frameShape = vid[0].shape

	SVs = np.transpose(np.array(pickle.load(open("glass_rose_xfer_sv.p", "r"))))

#	print SVs.shape

	rewardFunc = rewardFunctionMaker(SVs, 10, 1, 0.1)

	x0 = np.random.normal(0, 1, SVs.shape[1])

	resultVec = minimize(rewardFunc, x0, method='COBYLA', options={'maxiter':1000})['x']


	basisVec = np.dot(SVs, resultVec)
	viewFrame(imageify(np.reshape(basisVec, frameShape)), \
		adaptiveScaling=True, differenceImage=True)


#	print rewardFunc



