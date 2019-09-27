
import numpy as np
from import_1dify import fuzzyLookup
import matplotlib.pyplot as p
from video_magnifier import viewFrame, viewFrameR, viewFlatFrame
from image_distortion_simulator import imageify, imageifyComplex
from scipy.linalg import dft, circulant, hadamard, toeplitz, hankel
from blind_deconvolution import resizeArray1D
from math import tan, atan, log, pi, exp, sqrt

def makeTiltMatrix(n):
	returnArray = []

	returnArray.append([0]*n)

	for i in range(n-2):
		row = []
		l = [0]*(n-2)
		l[i] = 1
		print(l)

		for j in range(n):
			ls = np.linspace(0, n-3, n)

			row.append(fuzzyLookup(l, ls[j]))

		returnArray.append(row)

	returnArray.append([0]*n)

	return np.array(returnArray)
#		for j in range(n):

def makeSimpleSequence(n):
	return np.array(list(range(n)))/n

def makeSlightlyLessSimpleSequence(n):
	firstLegLength = int(n/6)
	secondLegLength = int(n/3)
	thirdLegLength = int(n/3)
	fourthLegLength = n - firstLegLength - secondLegLength - thirdLegLength

	return np.concatenate([makeSimpleSequence(firstLegLength), \
							makeSimpleSequence(secondLegLength)[::-1], \
							makeSimpleSequence(thirdLegLength), \
							makeSimpleSequence(fourthLegLength)[::-1]])

def makeSequenceIntoTiltMatrix(seq, startFraction=0.1):
	seqLength = len(seq)

	returnArray = []

	endFraction = seqLength

	fractions = np.linspace(startFraction, endFraction, seqLength)

	arctanFractions = np.linspace(1/(startFraction), 1/(endFraction), int(seqLength)/2)

	for arctanFraction in arctanFractions:
		fraction = 1/(arctanFraction)
		if fraction < 1:

			row = resizeArray1D(seq[int((0.5 - (fraction/2))*seqLength): \
				int((0.5 + (fraction/2))*seqLength)], \
				seqLength)

			returnArray.append(row)

		else:
			paddingSize = int(((fraction - 1)/2)*(seqLength/fraction))
			print(seqLength, paddingSize)

			innerStuffSize = seqLength - 2*paddingSize

			row = np.concatenate([[0]*paddingSize, \
				resizeArray1D(seq, innerStuffSize), [0]*paddingSize])

			returnArray.append(row)

	backwardsRows = []

	for row in returnArray[::-1]:
		backwardsRows.append(row[::-1])

	returnArray.extend(backwardsRows)

	p.matshow(np.array(returnArray), cmap="Greys_r")
	p.show()

	return returnArray

#	viewFlatFrame(imageify(firstRow))

def monomialEigenbasis(n):
	returnArray = []

	print(n)

	for k in range(n):
		returnArray.append([x**(2*pi*1j*k*100/n) for x in range(1,n+1)])

	return np.array(returnArray)

def fourierEigenbasis(n):
	returnArray = []

	for k in range(n):
		returnArray.append([np.exp(1j*2*pi*k*x/n) for x in range(n)])

	return np.array(returnArray)	

SIMPLE_SEQ = False
TILT = False
TILT_2 = True

if SIMPLE_SEQ:

#	print makeSimpleSequence(100)

	seq = makeSlightlyLessSimpleSequence(100)

	viewFlatFrame(imageify(seq))



	p.matshow(circulant(seq), cmap="Greys_r")
	p.show()

if TILT:

#	p.plot(sorted(np.linalg.eig(makeTiltMatrix(10))[0]))
#	p.show()

#	p.matshow(np.real(np.linalg.eig(makeTiltMatrix(10))[1]))
#	p.colorbar()
#	p.show()

	n = 1000

	T = makeTiltMatrix(n)
	E = np.transpose(T)

	p.matshow(np.linalg.matrix_power(T, int(n/2)))
	p.colorbar()
	p.show()

	p.matshow(T)
	p.colorbar()
	p.show()
	p.matshow(E)
	p.colorbar()
	p.show()
	p.matshow(np.dot(T, E) - np.identity(n))
	p.colorbar()
	p.show()
	p.matshow(np.dot(E, T) - np.identity(n))
	p.colorbar()
	p.show()

	p.matshow(np.dot(np.dot(np.dot(E, T), E), T) - np.identity(n))
	p.colorbar()
	p.show()

if TILT_2:
	n = 100

	seq = makeSlightlyLessSimpleSequence(n)
	tm = makeSequenceIntoTiltMatrix(seq, startFraction=0.3)

	u, s, v = np.linalg.svd(tm)

	p.matshow(np.real(u), cmap="Greys_r")
	p.colorbar()
	p.show()

	p.matshow(np.real(v), cmap="Greys_r")
	p.colorbar()
	p.show()

	meb = monomialEigenbasis(n)


	p.matshow(np.real(np.dot(np.dot(u, tm), np.conj(np.transpose(v)))))
	p.colorbar()
	p.show()

	p.matshow(np.real(np.dot(meb, np.conj(np.transpose(meb)))), cmap="Greys_r")
	p.colorbar()
	p.show()

#	tm = circulant(seq)
#	meb = fourierEigenbasis(n)

	p.matshow(np.real(meb), cmap="Greys_r")
	p.show()

	diag = np.dot(np.dot(meb, tm), np.transpose(np.conj(meb)))

	p.matshow(np.real(diag), cmap="Greys_r")
	p.colorbar()
	p.show()

	meb = dft(n)
	tm = circulant(seq)

	diag = np.dot(np.dot(meb, tm), np.transpose(np.conj(meb)))

	p.matshow(np.real(meb), cmap="Greys_r")
	p.show()

	p.matshow(np.real(diag), cmap="Greys_r")
	p.colorbar()
	p.show()


