import numpy as np
from math import sqrt, log
import matplotlib.pyplot as p
from video_magnifier import viewFrame, viewFlatFrame
from image_distortion_simulator import imageify

def dft(x):
	return np.fft.fft(x)

def idft(x):
	return np.fft.ifft(x)

def setToMags(x, mags):
#	print "xToBeSet", x
#	print "mags", mags

	return [i/np.abs(i)*m for i, m in zip(x, mags)]

def roundToBinary(x):
	binaryX = np.array([1*(i>0.5) for i in x])
	averageMSE = np.mean(np.multiply(x - binaryX, x - binaryX))

	return binaryX, averageMSE

def roundToBinarySoft(x, softness=1):
	softBinaryX = np.array([1*(i>1) + (softness*i+(1-softness)*(i>0.5))*((0<i) and (i<1)) for i in x])
	averageMSE = np.mean(np.multiply(x - softBinaryX, x - softBinaryX))

	return softBinaryX, averageMSE

def retrievePhase(mags, maxTries=100, giveUpLevel=1000, tol=1e-6):
	n = len(mags)

	tryCounter = 0
	foundSolution = False
	solution = None

	hardeningRate = 0.001
	hardeningRate = 0
	hardness = 0

	while tryCounter < maxTries and foundSolution == False:
		giveUpCounter = 0
		print tryCounter
		a = np.random.binomial(1, 0.5, n).astype(float)
		errors = []

		while giveUpCounter < giveUpLevel:
			giveUpCounter += 1
			dftA = dft(a)

			maggedDftA = setToMags(dftA, mags)
			maggedA = np.real(idft(maggedDftA)) # im part should be tiny

			a, error = roundToBinarySoft(maggedA, 1-hardness)

#			print a

			if error < tol:
				foundSolution = True
				solution = a

				break

			errors.append(error)

		tryCounter += 1
		hardness += hardeningRate

	if not foundSolution:
		solution = a
	return solution, foundSolution, errors

if False:

	n = 25
	truth = np.random.binomial(1, 0.5, n)
	print truth

	#a = np.random.binomial(1, 0.5, n).astype(float)
	rage = np.array([0]*n).astype(float)

	tol = 1e-6

	mags = np.abs(dft(truth))

	#print "mags", mags

	rageIsOn = False
	rageIncreaseRate = 0.0000001
	rageConstant = 0

	foundSolution = False
	solution = None
	outerCounter = 0

	while outerCounter < 100 and foundSolution == False:
		counter = 0
		print outerCounter
		a = np.random.binomial(1, 0.5, n).astype(float)
		errors = []

		while counter < 100:
			counter += 1

		#	print "a", a
		#	print "rage", rage
			if rageIsOn:
				adjustedA = a - rage*rageConstant
			else:
				adjustedA = a
			dftA = dft(adjustedA)
			maggedDftA = setToMags(dftA, mags)
			maggedA = np.real(idft(maggedDftA)) # im part should be tiny
		#	print "maggedA", maggedA
		#	print "sumMaggedA", sum(maggedA)

			rage += maggedA - a
			if rageIsOn:
				enragedA = maggedA + rage*rageConstant
			else:
				enragedA = maggedA
		#	print "enragedA", enragedA

			a, error = roundToBinarySoft(enragedA)
	#		a, error = roundToBinary(enragedA)
			if counter % 1 == 0:
		#		print "error", error
				errors.append(log(error))

			if error < tol:
				foundSolution = True
				solution = a

				break

			rageConstant += rageIncreaseRate	

		outerCounter += 1

	viewFlatFrame(imageify(truth))
	viewFlatFrame(imageify(a))

	p.plot(errors)
	p.show()

if True:
	n = 25
	truth = np.random.binomial(1, 0.5, n)

	print truth

	mags = np.abs(dft(truth)+np.random.normal(0,0.02))

	recoveredSeq, didWeSucceed, errors = retrievePhase(mags)

	print recoveredSeq

	print "Did we succeed?", didWeSucceed

	viewFlatFrame(imageify(truth))
	viewFlatFrame(imageify(recoveredSeq))
	p.plot(errors)
	p.show()