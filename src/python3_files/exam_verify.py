from math import log, floor

def padIntegerWithZeros(x, maxLength):
    if x == 0:
        return "0"*maxLength

    eps = 1e-8


    assert log(x+0.0001, 10) < maxLength

    return "0"*(maxLength-int(floor(log(x, 10)+eps))-1) + str(x)

def exactlyThree2s(x):
	num2s = 0

	for c in x:
		if c == "2":
			num2s += 1

	if num2s == 3:
		return True
	return False

def sumsToK(x, k):
	returnVal = 0

	for c in x:
		returnVal += int(c)

	if returnVal == k:
#		print(x)
		return True
	return False

def allDigitsLessThanOrEqualToK(x, k):
	for c in x:
		if int(c) > k:
			return False
	return True

def fac(n):
	if n <= 0:
		return 1
	else:
		return n * fac(n-1)

def choose(n, k):
	return fac(n)/(fac(n-k)*fac(k))

CHECK_EXAM = False
JEFF_PROBLEM = True

if CHECK_EXAM:

	count = 0
	n = 100000

	for i in range(n):
		paddedI = padIntegerWithZeros(i, 5)

		if exactlyThree2s(paddedI):
			count += 1

	print(count)

	count = 0

	for i in range(n):
		paddedI = padIntegerWithZeros(i, 5)

		if sumsToK(paddedI, 9):
			count += 1

	print(count)

	count = 0

	for i in range(n):
		paddedI = padIntegerWithZeros(i, 5)

		if exactlyThree2s(paddedI) or sumsToK(paddedI, 9):
			count += 1

	print(count)

if JEFF_PROBLEM:

	n = 100000

	maxVal = 9
	sumVal = 14
	numDigits = 5

	count = 0
	n = 10**numDigits	

	for i in range(int(n)):
		paddedI = padIntegerWithZeros(i, 5)

		if sumsToK(paddedI, sumVal) and allDigitsLessThanOrEqualToK(paddedI, maxVal):
			count += 1

	print("Actual count:", count)

	myRecitationGuess = choose(sumVal + numDigits - 1, numDigits - 1) - \
		choose(sumVal - maxVal + numDigits - 1, numDigits - 1)

	print("Recitation guess count:", myRecitationGuess)
