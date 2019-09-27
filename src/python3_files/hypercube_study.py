
from search import randomGreedySearchStepCount
import random
import string
import re
import matplotlib.pyplot as p

MONTE_CARLO_WORST_HYPERCUBE = False
GRAY_HYPERCUBE = True
RECOVER_BAD_HC = False

class Hypercube:
	def __init__(self, d, funcDict):
		self.funcDict = funcDict
		self.d = d
		if len(funcDict) == 2**d:
			self.isFull = True
		elif len(funcDict) < 2**d:
			self.isFull = False
		else:
			raise

		self.assembleInverseFuncDict()

	def assembleInverseFuncDict(self):
		self.inverseFuncDict = {}

		for key in self.funcDict:
			val = self.funcDict[key]
			self.inverseFuncDict[val] = key

	def evaluate(self, inp):
		return self.funcDict[inp]

	def evaluateFunc(self):
		return self.evaluate

	def getListOfNeighbors(self):
		returnDict = {}

		for vertex in self.funcDict:
			vertexNeighbors = getNeighborsOfTuple(vertex)

			for neighbor in vertexNeighbors:
				if not neighbor in self.funcDict:
					if not neighbor in returnDict:
						returnDict[neighbor] = True

		return list(returnDict.keys())

	def addEntry(self, key, val):
		self.funcDict[key] = val
		self.inverseFuncDict[val] = key

	def assignRandomNeighbor(self, val):
		neighbors = self.getListOfNeighbors()
#		print neighbors

		randomNeighbor = random.choice(neighbors)

		self.addEntry(randomNeighbor, val)

	def displayByDecreasingVal(self):
		for val in range(2**self.d - 1, -1, -1):
			print(self.inverseFuncDict[val], ":", val)

def getNeighborsOfTuple(t):
	returnList = []
	l = list(t)
	for i in range(len(t)):
		l[i] = 1 - l[i]
		returnList.append(tuple(l))
		l[i] = 1 - l[i]

	return returnList

def allTuplesOfSizeX(x):
    if x == 0:
        return [()]

    else:
        oneLess = allTuplesOfSizeX(x-1)
        return [i + tuple([0]) for i in oneLess] + [i + tuple([1]) for i in oneLess]

def makeRandomHypercube(d):
	randomPermutation = list(range(2**d))
	random.shuffle(randomPermutation)

	funcDict = {}

	for i, tup in zip(randomPermutation, allTuplesOfSizeX(d)):
		funcDict[tup] = i

	return Hypercube(d, funcDict)

def makeRandomOneMaxHypercube(d):
	hc = Hypercube(d, {tuple([1]*d): 2**d-1})

	for val in range(2**d - 2, -1, -1):
#		print val
		hc.assignRandomNeighbor(val)

	return hc

def binaryToGray(x):
	return x ^ (x >> 1)

def makeGrayCodeHypercube(d):
	funcDict = {}

	for i in range(2**d):
		grayCodeI = binaryToGray(i)

		funcDict[tuple([int(j) for j in ('{0:0' + str(d) + 'b}').format(grayCodeI)])] = i
#		funcDict[tuple([int(j) for j in ('{0:0' + str(d) + 'b}').format(i)])] = grayCodeI

	return Hypercube(d, funcDict)


def randomlyFillOneMaxHypercube(hc, maxVal):
	hcCopy = Hypercube(hc.d, hc.funcDict.copy())

	for val in range(maxVal - 1, -1, -1):

		hcCopy.assignRandomNeighbor(val)

	return hcCopy

def makeBadHypercube(d, numHCSamples=20):
	masterHC = Hypercube(d, {tuple([1]*d): 2**d-1})

	for val in range(2**d - 2, -1, -1):
		bestRunningSum = 0
		bestNeighbor = None

		for neighbor in masterHC.getListOfNeighbors():
			hcCopy = Hypercube(masterHC.d, masterHC.funcDict.copy())
			hcCopy.addEntry(neighbor, val)

			runningSum = 0

			for _ in range(numHCSamples):
				filledHC = randomlyFillOneMaxHypercube(hcCopy, val)

				runningSum += stepCountSample(filledHC)

			runningSum /= numHCSamples

			if runningSum > bestRunningSum:
				bestRunningSum = runningSum
				bestNeighbor = neighbor

		print(bestRunningSum)

		masterHC.addEntry(bestNeighbor, val)

	return masterHC

def stepCountSample(hc, numSamples=200):
	evalFunc = lambda x: (hc.evaluate(tuple(x)), None)

	runningSum = 0

	for _ in range(numSamples):
		init = [(random.random()<0.5)*1 for _ in range(hc.d)]

		stepCount = randomGreedySearchStepCount(init, evalFunc, maxOrMin="max")

#		print stepCount

		runningSum += stepCount

	return runningSum/numSamples

def recoverFuncDictFromFile(path):
	inp = open(path, "r").read()

	funcDict = {}

	inpEntries = re.compile(":|\(|\)").split(inp)

	i = 1

	while i+2 < len(inpEntries):
		rawKey = inpEntries[i]
		keyElts = string.split(rawKey, ",")

		keyList = []
		for elt in keyElts:
			keyList.append(int(elt))

		key = tuple(keyList)

		rawVal = inpEntries[i+2]

		if rawVal[-1] == "}":
			val = int(rawVal[1:-1])
		else:
			val = int(rawVal[1:-2])

		funcDict[key] = val

		print(key, val)

		i += 3

	print(len(funcDict))

	return funcDict

def recoverPlotFromFile(path):
	inp = open(path, "r").readlines()

	yVals = []

	for entry in inp:
		yVals.append(float(entry))

	p.plot(yVals)
	p.axhline(y=7.4, color="k")
	p.show()

if MONTE_CARLO_WORST_HYPERCUBE:
	d = 6

#	hc = makeRandomOneMaxHypercube(d)
#	hc = makeRandomHypercube(d)

	print(makeBadHypercube(d).funcDict)

if GRAY_HYPERCUBE:
	d = 6

	hc = makeGrayCodeHypercube(d)
	hc.displayByDecreasingVal()

	print(stepCountSample(hc, numSamples=2000))

if RECOVER_BAD_HC:
	funcDict = recoverFuncDictFromFile("/Users/adamyedidia/bad_hypercube.txt")
	recoverPlotFromFile("/Users/adamyedidia/bad_hypercube_vals.txt")
	d = 6

	hc = Hypercube(d, funcDict)

	hc.displayByDecreasingVal()

