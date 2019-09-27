
import numpy as np
from math import sqrt, log, cos, acos, pi
import matplotlib.pyplot as p
from video_magnifier import viewFrame, viewFlatFrame
from image_distortion_simulator import imageify, imageifyComplex, circleSpeck, getR, \
	doFuncToEachChannel, circleSpeck, squareSpeck, batchAndDifferentiate
from video_processor import batchArrayAlongZeroAxis
import random
from lasso_ista_fista import ista, fista
from scipy.linalg import dft as dftMat
import pickle
from best_matrix import padIntegerWithZeros
from PIL import Image

PHASE_RETRIEVAL = False
PHASE_RETRIEVAL_2D = False
DIVIDE_AND_CONQUER = False
CHRISTOS_INFER_PHASES_TEST = False
CHRISTOS_INFER_PHASES = False
PHASE_RETRIEVAL_2D_LARGE = False
PHASE_RETRIEVAL_CVPR = False
VIEW_NAT_IMAGE_PHASE_ONLY = False
VIEW_NAT_IMAGE_MAG_ONLY = False
WINDOW_VID = False
PHASE_VID = False
PHASE_DIFF = False
WINDOW_IM = True

def cis(val):
    angle = np.angle(val, deg=False)

    if angle >= 0:
        return angle
    else:
        return angle + 2*pi

def putBetweenNegativePiAndPi(val):
#	print val
	if val > pi:
		return putBetweenNegativePiAndPi(val - 2*pi)
	elif val < -pi:
		return putBetweenNegativePiAndPi(val + 2*pi)
	else:
		return val

def majorityGame(arr, coolingTime):
    maxX = arr.shape[0]
    maxY = arr.shape[1]

    for i in range(coolingTime):
        newArr = np.zeros(arr.shape)

        print(i)

        for x in range(-1, maxX-1):
            for y in range(-1, maxY-1):
                voteCounter = 0
                for deltaX in [-1, 0, 1]:
                    for deltaY in [-1, 0, 1]:
                        voteCounter += arr[x+deltaX][y+deltaY]
                if voteCounter > 4:
                    newArr[x][y] = 1

        arr = newArr
    return arr

def hanning2D(shape):
	assert len(shape) == 2

	return np.dot(np.reshape(np.hanning(shape[0]), (shape[0], 1)), \
		np.reshape(np.hanning(shape[1]), (1, shape[1])))

def windowFrame(frame):
	frameDims = frame.shape[:-1]

#	viewFrame(imageify(hanning2D(windowDims)))

	return np.multiply(imageify(hanning2D(frameDims)), frame)

def generateRandomSparseOccluder(x, y):
	onOffArr = np.random.binomial(1, 0.1, (x, y))

	valArr = np.random.exponential(1, (x, y))

	return np.multiply(onOffArr, valArr)

def generateRandomCorrelatedOccluder(x, y):
    arr = np.random.binomial(1, 0.5, (x, y))

    return majorityGame(arr, 10)

def convertArrToVec(x):
	return x.flatten()

def convertVecToArrMaker(shape):
	def convertVecToArr(x):
		return np.reshape(x, shape)
	return convertVecToArr

def dft(x):
	return np.fft.fft(x)

def idft(x):
	return np.fft.ifft(x)

def putInRange(angle, zeroToTwoPi=True):
	if zeroToTwoPi:
		return angle % (2*pi)
	else:
		zeroToTwoPiAngle = angle % (2*pi)
		if zeroToTwoPiAngle > pi:
			return zeroToTwoPiAngle - 2*pi
		else:
			return zeroToTwoPiAngle

def setToMags(x, mags):
#	print "xToBeSet", x
#	print "mags", mags

	returnList = []

	for i, m in zip(x, mags):
		if i == 0:
			returnList.append(m)
		else:
			returnList.append(i/np.abs(i)*m)

	return returnList

def setToMags2D(x, mags):
	return np.multiply(np.divide(x, np.abs(x)), mags)

def roundEltToBinarySoft(x):
	if x > 1:
		return 1
	elif x < 0:
		return 0
	else:
		return x

def roundToBinarySoft2D(x):
	roundedX = np.vectorize(roundEltToBinarySoft)(x)
	averageMSE = np.mean(np.multiply(x - roundedX, x - roundedX))

	return roundedX, averageMSE

def sparsify(arr, cutoffFactor=1):
    averageVal = np.sum(np.abs(arr))/np.size(arr)

#    print averageVal

    biggerThanCutoff = np.vectorize(lambda x: 1*(x>averageVal/cutoffFactor))

#    viewFrame(imageify(biggerThanCutoff(np.abs(arr))), adaptiveScaling=True, \
 #       differenceImage=True)

    result = np.multiply(arr, biggerThanCutoff(np.abs(arr)))

    diff = result - arr

    return result, np.sum(np.multiply(diff, diff))

def roundToBinary(x):
	binaryX = np.array([1*(i>0.5) for i in x])
	averageMSE = np.mean(np.multiply(x - binaryX, x - binaryX))

	return binaryX, averageMSE

def roundToBinarySoft(x, softness=1):
	softBinaryX = np.array([1*(i>1) + (softness*i+(1-softness)*(i>0.5))*((0<i) and (i<1)) for i in x])
	averageMSE = np.mean(np.multiply(x - softBinaryX, x - softBinaryX))

	return softBinaryX, averageMSE

def retrievePhase2D(mags, primalDomainFunc, maxTries=100, giveUpLevel=100, tol=1e-6,
	initializationFunc=None):

	n = len(mags)

	tryCounter = 0
	foundSolution = False
	solution = None

	bestSolutionSoFar = None
	bestErrorSoFar = float("Inf")
	bestListOfErrorsSoFar = None



	while tryCounter < maxTries and foundSolution == False:
		giveUpCounter = 0
		print(tryCounter)
		if initializationFunc == None:
			initialization = np.random.binomial(1, 0.5, mags.shape).astype(float)
		else:
			initialization = initializationFunc()
	
		a = initialization
		errors = []

		while giveUpCounter < giveUpLevel:
			oldA = a

			giveUpCounter += 1
			dftA = np.fft.fft2(a)

			maggedDftA = setToMags2D(dftA, mags)
			maggedA = np.real(np.fft.ifft2(maggedDftA)) # im part should be tiny



#			print "magged A", maggedA
#			viewFrame(imageify(maggedA), adaptiveScaling=True)

			a, _ = primalDomainFunc(maggedA)

#			print "fista'd A", a
#			viewFrame(imageify(a), adaptiveScaling=True)

#			print a

#			print a, oldA

			diff = a - oldA

			error = np.sum(np.multiply(diff, diff))

			print("error", error)

			if error < tol:
				foundSolution = True
				solution = a

				break


			errors.append(error)

		if error < bestErrorSoFar:
			bestSolutionSoFar = a
			bestErrorSoFar = error
			bestListOfErrorsSoFar = errors

		tryCounter += 1

	if not foundSolution:
		solution = bestSolutionSoFar
		errors = bestListOfErrorsSoFar

	return solution, foundSolution, errors, initialization

def retrievePhaseOneTry(mags, giveUpLevel=100, tol=1e-6, initialization=None):
	if initialization == None:
		a = np.random.binomial(1, 0.5, n).astype(float)
	else:
		a = initialization

	giveUpCounter = 0
	errors = []
	foundSolution = False

	while giveUpCounter < giveUpLevel:
		print(a)

		giveUpCounter += 1
		dftA = dft(a)

#		print giveUpCounter

		maggedDftA = setToMags(dftA, mags)
		maggedA = np.real(idft(maggedDftA)) # im part should be tiny

		a, error = roundToBinarySoft(maggedA, 1)

#			print a

		if error < tol:
			foundSolution = True
			solution = a

			break

		errors.append(error)

	return a, foundSolution, errors

def retrievePhase(mags, maxTries=100, giveUpLevel=100, tol=1e-6):
	n = len(mags)

	tryCounter = 0
	foundSolution = False
	solution = None

#	hardeningRate = 0.001
	hardeningRate = 0
	hardness = 0

	while tryCounter < maxTries and foundSolution == False:
		print(tryCounter)

		solution, foundSolution, errors = retrievePhaseOneTry(mags, giveUpLevel, tol)

		tryCounter += 1
		hardness += hardeningRate

	return solution, foundSolution, errors

def inferPhaseFromPreviousPhasesSingleEntry(firstMag, secondMag, thirdMag, 
	diff31Mag, diff32Mag, firstPhase, secondPhase):

	eps = 1e-10
#	eps = 0

#	firstMag += eps
#	secondMag += eps
#	thirdMag += eps

	firstAcosVal = (thirdMag**2 + firstMag**2 - \
		diff31Mag**2)/(2*thirdMag*firstMag)

	secondAcosVal = (thirdMag**2 + secondMag**2 - \
		diff32Mag**2)/(2*thirdMag*secondMag)

	if firstAcosVal > 1:
		absPhaseDiff31 = 0 
		print("Warning: rounded", firstAcosVal, "down to 1")

	elif firstAcosVal < -1:
		print("Warning: rounded", firstAcosVal, "up to -1")
		absPhaseDiff31 = pi

	else:
		absPhaseDiff31 = acos((thirdMag**2 + firstMag**2 - \
			diff31Mag**2)/(2*thirdMag*firstMag))		

	if secondAcosVal > 1:
		print("Warning: rounded", secondAcosVal, "down to 1")
		absPhaseDiff32 = 0

	elif secondAcosVal < -1:
		print("Warning: rounded", secondAcosVal, "up to -1")		
		absPhaseDiff32 = pi

	else:
		absPhaseDiff32 = acos((thirdMag**2 + secondMag**2 - \
			diff32Mag**2)/(2*thirdMag*secondMag))	

	possiblePhases31 = [putInRange(firstPhase - absPhaseDiff31),
						putInRange(firstPhase + absPhaseDiff31)]

	possiblePhases32 = [putInRange(secondPhase - absPhaseDiff32),
						putInRange(secondPhase + absPhaseDiff32)]	

	smallestDifference = float("Inf")
	bestPhase = None

#	print possiblePhases31, possiblePhases32

	for phase31 in possiblePhases31:
		for phase32 in possiblePhases32:

			phaseDifference = putInRange(phase31 - phase32, zeroToTwoPi=False)

			if abs(phaseDifference) < smallestDifference:
				if smallestDifference < eps:
					print("Warning: ambiguous choice", possiblePhases31, possiblePhases32)
				smallestDifference = abs(phase31 - phase32)
				bestPhase = putInRange((phase31 + phase32)/2)

	if smallestDifference > eps:
		print("Warning: large difference", smallestDifference)

#	print bestPhase
#	print ""

	return thirdMag*np.exp(1j*bestPhase)

def inferPhaseFromPreviousPhasesOld(firstMags, secondMags, thirdMags, 
	diff31Mags, diff32Mags, firstPhases, secondPhases):
		
	arrShape = firstMags.shape	
	maxX = arrShape[0]
	maxY = arrShape[1]

	returnArray = []

	for i in range(maxX):
		returnArray.append([])
		for j in range(maxY):
			firstMag = firstMags[i][j]
			secondMag = secondMags[i][j]
			thirdMag = thirdMags[i][j]
			diff31Mag = diff31Mags[i][j]
			diff32Mag = diff32Mags[i][j]
			firstPhase = firstPhases[i][j]
			secondPhase = secondPhases[i][j]

			returnArray[-1].append(inferPhaseFromPreviousPhasesSingleEntry(firstMag,
				secondMag, thirdMag, diff31Mag, diff32Mag, firstPhase, secondPhase))

	return np.array(returnArray)

def inferPhaseFromPreviousPhases(firstMags, secondMags, thirdMags, 
	diff31Mags, diff32Mags, firstPhases, secondPhases):

	return np.vectorize(inferPhaseFromPreviousPhasesSingleEntry)(firstMags, \
		secondMags, thirdMags, diff31Mags, diff32Mags, firstPhases, secondPhases)

def getMags(arr):
	return np.abs(np.fft.fft2(arr))

def getPhases(arr):
	return np.vectorize(cis)(np.fft.fft2(arr))

def assembleComplexFromMagAndPhase(mag, phase):
	return mag*np.exp(1j*phase)

def getPhaseOnlyVersion(arr):
	phases = np.vectorize(cis)(np.fft.fft2(arr))

	viewFrame(imageify(phases), adaptiveScaling=True)

	mags = np.ones(arr.shape)
#	mags = np.random.random(arr.shape)

	return np.fft.ifft2(np.vectorize(assembleComplexFromMagAndPhase)(mags, phases))

def convertOccToZeroOne(occ):

    averageVal = np.sum(occ)/np.size(occ)

    candidateOccs = []

    for logOffset in np.linspace(-0.5, 0.5, 30):

        offset = 10**logOffset

        candidateOccs.append(np.vectorize(lambda x: 1.0*(x > offset*averageVal))(occ))

    return candidateOccs


def getPhases(arr):
	return np.vectorize(cis)(np.fft.fft2(arr))

def getEarlyPhaseOnlyVersion(arr, tol):
#	arr = squareSpeck((28, 28), 1/5)

	phases = np.vectorize(cis)(np.fft.fft2(arr))
#	randomPhases = 2*pi*np.random.random(arr.shape)
	randomPhases = np.zeros(arr.shape)



	def isEarlyMaker(maxI, maxJ, tol):
#		print tol
		def isEarly(i, j):
			if i < tol or maxI - i < tol:
				return 1
			elif j < tol or maxJ - j < tol:
				return 1
			else:
				return 0

		return isEarly

	def mux(a, b, muxVal):
		if muxVal == 0:
			return a
		elif muxVal == 1:
			return b
		else:
			raise

	mags = np.abs(np.fft.fft2(arr))
#	mags = np.ones(arr.shape)

#	viewFrame(imageify(arr), adaptiveScaling=True)

#	print "phases"
#	print phases
	viewFrame(imageify(phases), adaptiveScaling=True, colorbar=True)
#	viewFrame(imageify(randomPhases), adaptiveScaling=True)

	meshX, meshY = np.meshgrid(np.array(list(range(arr.shape[0]))), \
		np.array(list(range(arr.shape[1]))))

	earlyPhaseOnes = np.transpose(np.vectorize(isEarlyMaker(arr.shape[0], \
		arr.shape[1], tol))(meshX, meshY))
#	earlyPhaseOnes = np.transpose(np.vectorize(lambda i,j: 1-isEarly(i,j))(meshX, meshY))
		
#	viewFrame(imageify(earlyPhaseOnes), adaptiveScaling=True)

	phasesEarlyOnly = np.vectorize(mux)(randomPhases, phases, earlyPhaseOnes)

#	viewFrame(imageify(phasesEarlyOnly), adaptiveScaling=True)

	returnArray = np.fft.ifft2(np.vectorize(assembleComplexFromMagAndPhase)(mags, phasesEarlyOnly))

#	viewFrame(imageify(returnArray), adaptiveScaling=True)

	return returnArray

def getMagOnlyVersion(arr):
	mags = np.abs(np.fft.fft2(arr))

	phases = np.zeros(arr.shape)
#	phases = 2*pi*np.random.random(arr.shape)

#	print phases

	return np.fft.ifft2(np.vectorize(assembleComplexFromMagAndPhase)(mags, phases))

if __name__ == "__main__":

	if PHASE_RETRIEVAL:
		n = 25
		truth = np.random.binomial(1, 0.5, n)

		print(truth)

		mags = np.abs(dft(truth)+np.random.normal(0,0.0))

		recoveredSeq, didWeSucceed, errors = retrievePhase(mags)

		print(recoveredSeq)

		print("Did we succeed?", didWeSucceed)

		print("Ground truth")
		viewFlatFrame(imageify(truth))

		print("our recovery")
		viewFlatFrame(imageify(recoveredSeq))
		p.plot(errors)
		p.show()

	if PHASE_RETRIEVAL_2D:
		n = 5

	#	truth = np.random.binomial(1, 0.5, (n, n))

	#	truth = np.array(
	#		[[0,0,0,0,0,0,0],
	#		 [0,0,0,0,0,0,0],
	#		 [0,0,0,1,0,0,0],
	#		 [0,0,1,1,1,0,0],
	#		 [0,0,1,1,0,0,0],
	#	     [0,0,0,1,0,0,0],
	#		 [0,0,0,0,0,0,0]])

	#	truth = np.array(
	#		[[0,0,0,0,0,0,0],
	#		 [0,0,0,0,0,0,0],
	#		 [0,0,0,1,0,0,0],
	#		 [0,0,0,0,0,0,0],
	#		 [0,0,0,0,0,0,0],
	#	     [0,0,1,1,0,0,0],
	#		 [0,0,0,0,0,0,0]])

	#	truth = generateRandomCorrelatedOccluder(n,n)


	#	truth = circleSpeck((n, n), 0.10)*n**2

		truth = generateRandomSparseOccluder(n, n)

		doubleDft = np.kron(dftMat(n), dftMat(n))

		print(truth)
		viewFrame(imageify(truth))

		mags = np.abs(np.fft.fft2(truth)+np.random.normal(0,0.0))

		convertVecToArr = convertVecToArrMaker((n, n))

		print(mags)

	#	recoveredSeq, didWeSucceed, errors, initialization = retrievePhase2D(mags, \
	#		lambda x: sparsify(x, 0.4))

		def initializationFunc():
			return generateRandomSparseOccluder(n, n)

		def primalDomainFunc(x):
			I = np.identity(n*n)

			newVec = fista(I, convertArrToVec(x), 0.1, 100)[0]
			diff = newVec - convertArrToVec(x)
			error = np.sum(np.multiply(diff, diff))

	#		print "error", error

	#		viewFrame(imageify(x))

	#		viewFrame(imageify(convertVecToArr(newVec)))

			return convertVecToArr(newVec), error

		def initializationFuncTrivial():
			return truth

		recoveredSeq, didWeSucceed, errors, initialization = retrievePhase2D(mags, \
			primalDomainFunc, initializationFunc=initializationFunc)

		print(recoveredSeq)

		print("Did we succeed?", didWeSucceed)

		print("ground truth")
		viewFrame(imageify(truth))
		print("our recovery")
		viewFrame(imageify(recoveredSeq))
		print("initialization used")
		viewFrame(imageify(initialization))
		print("true magnitudes")
		viewFrame(imageify(mags), adaptiveScaling=True, magnification=1)
		print("recovered magnitudes")
		viewFrame(imageify(np.abs(np.fft.fft2(recoveredSeq))), adaptiveScaling=True, \
			magnification=1)

		print("error over time")
		p.plot(errors[1:])
		p.show()

	if DIVIDE_AND_CONQUER:
		n = 64
		logN = 6

		truth = np.random.binomial(1, 0.5, n)

		prevSolution = np.array([0.5])

		for logK in range(logN-1, -1, -1):
			batchSize = 2**logK
			print(batchSize)

			fuzzyTruth = batchArrayAlongZeroAxis(truth, batchSize)

			viewFlatFrame(imageify(fuzzyTruth))

			fuzzyMags = np.abs(np.fft.fft(fuzzyTruth))

			repeatedPrevSolution = np.repeat(prevSolution, 2)

			prevSolution, foundSolution, errors = retrievePhaseOneTry(fuzzyMags, \
				initialization=repeatedPrevSolution)

			print(foundSolution)

			p.plot(errors)
			p.show()

			viewFlatFrame(imageify(prevSolution))

	if CHRISTOS_INFER_PHASES_TEST:
		x1 = random.random() + 1j*random.random()
		x2 = random.random() + 1j*random.random()

		m1 = np.abs(x1)
		m2 = np.abs(x2)

		phi1 = cis(x1)
		phi2 = cis(x2)

		print(np.real(x1*np.conj(x2)))
		print(m1*m2*cos(phi1-phi2))
		print("")
		print(np.abs(x1-x2)**2)
		print(np.abs(x1)**2 + np.abs(x2)**2 - 2*m1*m2*cos(phi2-phi1))
		print("")
		print(abs(phi1-phi2))
		print(acos((m1**2 + m2**2 - np.abs(x1-x2)**2)/(2*m1*m2)))

	if CHRISTOS_INFER_PHASES:
		for _ in range(100):

			n = 5
			firstFrame = np.random.binomial(1, 0.5, (n, n)) 
			secondFrame = np.random.binomial(1, 0.5, (n, n))
			thirdFrame = np.random.binomial(1, 0.5, (n, n))
			diff31Frame = thirdFrame - firstFrame
			diff32Frame = thirdFrame - secondFrame

			firstDFT = np.fft.fft2(firstFrame)
			secondDFT = np.fft.fft2(secondFrame)
			thirdDFT = np.fft.fft2(thirdFrame)
			diff31DFT = np.fft.fft2(diff31Frame)
			diff32DFT = np.fft.fft2(diff32Frame)

			firstMags = np.abs(firstDFT)
			secondMags = np.abs(secondDFT)
			thirdMags = np.abs(thirdDFT)
			diff31Mags = np.abs(diff31DFT)
			diff32Mags = np.abs(diff32DFT)

			firstPhase = np.vectorize(cis)(firstDFT)
			secondPhase = np.vectorize(cis)(secondDFT)
			thirdPhase = np.vectorize(cis)(thirdDFT)

			recoveredDFT = inferPhaseFromPreviousPhases(firstMags, secondMags, \
				thirdMags, diff31Mags, diff32Mags, firstPhase, secondPhase)

		#	print np.abs(recoveredDFT)
		#	print thirdMags

		#	print recoveredDFT
		#	print thirdDFT

			recoveredVal = np.fft.ifft2(recoveredDFT)
			print(recoveredVal)
			print(thirdFrame)

			viewFrame(imageify(np.real(recoveredVal)), differenceImage=False)
	#		viewFrame(imageify(thirdFrame), differenceImage=False)

	if PHASE_RETRIEVAL_2D_LARGE:
		vid = pickle.load(open("smaller_movie_batched_diff_framesplit.p", "r"))

		frameDims = [90,160]



	#	truth = circleSpeck((n, n), 0.10)*n**2

		truth = vid[200]

	#	doubleDft = np.kron(dftMat(n), dftMat(n))

		viewFrame(imageify(truth), adaptiveScaling=True)

		mags = np.abs(np.fft.fft2(truth)+np.random.normal(0,0.0))

		convertVecToArr = convertVecToArrMaker(frameDims)

		print(mags)

	#	recoveredSeq, didWeSucceed, errors, initialization = retrievePhase2D(mags, \
	#		lambda x: sparsify(x, 0.4))

		def initializationFunc():
			return generateRandomSparseOccluder(frameDims[0], frameDims[1])

		def primalDomainFunc(x):
			I = np.identity(frameDims[0]*frameDims[1])

			newVec = fista(I, convertArrToVec(x), 0.1, 100)[0]
			diff = newVec - convertArrToVec(x)
			error = np.sum(np.multiply(diff, diff))

	#		print "error", error

	#		viewFrame(imageify(x))

	#		viewFrame(imageify(convertVecToArr(newVec)))

			return convertVecToArr(newVec), error

		def initializationFuncTrivial():
			return truth

		recoveredSeq, didWeSucceed, errors, initialization = retrievePhase2D(mags, \
			primalDomainFunc, initializationFunc=initializationFunc, maxTries=1, giveUpLevel=1000,
			tol=1e-3)

		print(recoveredSeq)

		print("Did we succeed?", didWeSucceed)

		print("ground truth")
		viewFrame(imageify(truth))
		print("our recovery")
		viewFrame(imageify(recoveredSeq))
		print("initialization used")
		viewFrame(imageify(initialization))
		print("true magnitudes")
		viewFrame(imageify(mags), adaptiveScaling=True, magnification=1)
		print("recovered magnitudes")
		viewFrame(imageify(np.abs(np.fft.fft2(recoveredSeq))), adaptiveScaling=True, \
			magnification=1)

		print("error over time")
		p.plot(errors[1:])
		p.show()

	if PHASE_RETRIEVAL_CVPR:
		occ = pickle.load(open("corr_occ_2.p", "r"))

		mags = np.abs(np.fft.fft2(occ)+np.random.normal(0,0.1))	

		print(mags.shape)

		def primalDomainFunc(x):
			I = np.identity(x.shape[0]*x.shape[1])

			newVec = fista(I, convertArrToVec(x), 0.1, 100)[0]
			diff = newVec - convertArrToVec(x)
			error = np.sum(np.multiply(diff, diff))

	#		print "error", error

	#		viewFrame(imageify(x))

	#		viewFrame(imageify(convertVecToArr(newVec)))

			return np.reshape(newVec, x.shape), error

		arr = retrievePhase2D(mags, primalDomainFunc, maxTries=1, giveUpLevel=100, tol=1e-6,
			initializationFunc=None)[0]

		pickle.dump(arr, open("phase_retrieve_cvpr.p", "w"))

		viewFrame(imageify(arr))

	if VIEW_NAT_IMAGE_PHASE_ONLY:
#		im = pickle.load(open("dora_slightly_downsampled.p", "r"))
#		im = pickle.load(open("darpa_vid_extracted_occ.p", "r"))

#		im = np.array(Image.open("life_sized_velociraptor.png")).astype(float)

#		im = circleSpeck((28, 28), 1/5)
#		im = squareSpeck((28, 28), 1/5)
		im = generateRandomCorrelatedOccluder(28, 28)

		viewFrame(imageify(im))
#		viewFrame(im)

	#	viewFrame(imageify(getPhases(getR(im))), adaptiveScaling=True)
	#	maggedIm = doFuncToEachChannel(getMagOnlyVersion, im)
	#	phasedIm = doFuncToEachChannel(getPhaseOnlyVersion, im)


		for tol in range(50):
			print(tol)

#			maggedImWithEarlyPhase = doFuncToEachChannel(lambda x: \
#				getEarlyPhaseOnlyVersion(x, tol), im)

#			maggedImWithEarlyPhase = \
#				getEarlyPhaseOnlyVersion(im, tol)

#			print maggedImWithEarlyPhase

			print("magged")
			viewFrame(imageify(maggedImWithEarlyPhase), adaptiveScaling=True, \
				filename="phased_movie/frame_" + padIntegerWithZeros(tol, \
				3) + ".png", magnification=3, secondBiggest=True)

#			viewFrame(imageify(maggedImWithEarlyPhase), adaptiveScaling=True, \
#				magnification=3, secondBiggest=True)



	#	print phasedIm[0][0]
	#	print phasedIm[1][1]

#		viewFrame(maggedImWithEarlyPhase, adaptiveScaling=True, magnification=1)

	if VIEW_NAT_IMAGE_MAG_ONLY:
		im = pickle.load(open("dora_slightly_downsampled.p", "r"))

		maggedIm = doFuncToEachChannel(getMagOnlyVersion, im)

		viewFrame(maggedIm, adaptiveScaling=True, magnification=10)	

	if WINDOW_VID:
		vidName = "office_batched"

		vid = pickle.load(open(vidName + ".p"))

		windowedVid = []

		for i,frame in enumerate(vid):
			print(i)

			windowedFrame = windowFrame(frame)

			windowedVid.append(windowedFrame)

		pickle.dump(np.array(windowedVid), open(vidName + "_windowed.p", "w"))

	if WINDOW_IM:
		imName = "dora_28_28"

		im = pickle.load(open(imName + ".p"))

		windowedIm = windowFrame(im)

		pickle.dump(np.array(windowedIm), open(imName + "_windowed.p", "w"))

	if PHASE_VID:
		vidName = "office_batched_windowed"

		vid = pickle.load(open(vidName + ".p"))

		phasedVid = []

		for i,frame in enumerate(vid):
			print(i)

			phasedIm = getPhases(frame)

			phasedVid.append(phasedIm)

		pickle.dump(np.array(phasedVid), open(vidName + "_phased.p", "w"))

	if PHASE_DIFF:
		vidName = "office_batched_phased_phasediff"

		vid = pickle.load(open(vidName + ".p"))

		diffVid = batchAndDifferentiate(vid, [(1, False), (1, True), (1, True), \
			(1, False)])

		diffVidSlotted = np.vectorize(putBetweenNegativePiAndPi)(diffVid)

		viewFrame(vid[30], adaptiveScaling=True, differenceImage=False, \
			magnification=1)
		viewFrame(diffVidSlotted[30], adaptiveScaling=True, differenceImage=True, \
			magnification=1)

		pickle.dump(np.array(diffVidSlotted), open(vidName + "_phasediff.p", "w"))



