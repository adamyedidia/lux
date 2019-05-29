import matplotlib.pyplot as p
import pickle
import numpy as np
from math import log, floor, sqrt

BASIC_TEST = False
EXTRA_PLOT = False
FFT2D_TESTS = True

def getMags(arr):
    return np.abs(np.fft.fft2(arr))


def padIntegerWithZeros(x, maxLength):
    if x == 0:
        return "0"*maxLength

    eps = 1e-8


    assert log(x+0.0001, 10) < maxLength

    return "0"*(maxLength-int(floor(log(x, 10)+eps))-1) + str(x)

if __name__ == "__main__":
	if BASIC_TEST:

		correctDora = pickle.load(open("dora_28_28.p", "rb"))/255
		attemptedDora = np.reshape(pickle.load(open("mags/dora_100.p", "rb")), (28, 28))#*255

		p.matshow(correctDora, cmap="Greys_r")
		p.colorbar()
		p.show()

		p.matshow(attemptedDora, cmap="Greys_r")
		p.colorbar()
		p.show()

		correctDoraMags = getMags(correctDora)
		correctDoraMags[0][0] = 0

		attemptedDoraMags = getMags(attemptedDora)
		attemptedDoraMags[0][0] = 0

		print(attemptedDora.shape)

		p.matshow(correctDoraMags, cmap="Greys_r")
		p.colorbar()
		p.show()

		p.matshow(attemptedDoraMags, cmap="Greys_r")
		p.colorbar()
		p.show()

	if EXTRA_PLOT:
		correctDora = (pickle.load(open("dora_28_28.p", "rb")) - 127.5*np.ones((28, 28)))/127.5

		maxN = 24

#		print(correctDora)

		for i in range(maxN):
#			attemptedDora = 2*(np.reshape(pickle.load(open("mags/dora_" + padIntegerWithZeros(i, 3) \
#				+ ".p", "rb")), (28, 28)) - np.ones((28,28)))#*255			

			attemptedDora = np.reshape(pickle.load(open("mags/dora_" + padIntegerWithZeros(i, 3) + \
				".p", "rb")), (28, 28))

			if i == 0:
#				print(attemptedDora)

#				print("-------------")
				pass

			correctDoraMags = getMags(correctDora)
			attemptedDoraMags = getMags(attemptedDora)

			if i == 7:
#				print(correctDoraMags)
#				print(attemptedDoraMags)

				p.matshow(correctDoraMags, cmap="Greys_r")
				p.colorbar()
				p.show()

				p.matshow(attemptedDoraMags, cmap="Greys_r")
				p.colorbar()
				p.show()

			magDiff = correctDoraMags - attemptedDoraMags

#			print(magDiff)

			print(i, np.sum(np.multiply(magDiff, magDiff))/784)

	if EXTRA_PLOT:
		correctDora = (pickle.load(open("dora_28_28.p", "rb")) - 127.5*np.ones((28, 28)))/127.5

		maxN = 24

#		print(correctDora)

		for i in range(maxN):
#			attemptedDora = 2*(np.reshape(pickle.load(open("mags/dora_" + padIntegerWithZeros(i, 3) \
#				+ ".p", "rb")), (28, 28)) - np.ones((28,28)))#*255			

			attemptedDora = np.reshape(pickle.load(open("mags/dora_" + padIntegerWithZeros(i, 3) + \
				".p", "rb")), (28, 28))

			if i == 0:
#				print(attemptedDora)

#				print("-------------")
				pass

			correctDoraMags = getMags(correctDora)
			attemptedDoraMags = getMags(attemptedDora)

			if i == 7:
#				print(correctDoraMags)
#				print(attemptedDoraMags)

				p.matshow(correctDoraMags, cmap="Greys_r")
				p.colorbar()
				p.show()

				p.matshow(attemptedDoraMags, cmap="Greys_r")
				p.colorbar()
				p.show()

			magDiff = correctDoraMags - attemptedDoraMags

#			print(magDiff)

			print(i, np.sum(np.multiply(magDiff, magDiff))/784)

	if FFT2D_TESTS:
		direc = "mags/"
		moniker = "dora"

		epoch = 3000

		print(direc + moniker + "_img_" + \
            padIntegerWithZeros(int(epoch/200), 3) + ".p") 

		preFFT2D = pickle.load(open(direc + moniker + "_img_" + \
            padIntegerWithZeros(int(epoch/200), 3) + ".p", "rb"))

		attemptedPostFFT2D = np.fft.fft2(preFFT2D)/sqrt(6)

		attemptedPostAbsFFT2D = np.abs(np.fft.fft2(preFFT2D))/sqrt(6)

		postFFT2D = pickle.load(open(direc + moniker + "_" + \
            padIntegerWithZeros(int(epoch/200), 3) + ".p", "rb"))

		print(preFFT2D)
		print(attemptedPostFFT2D)
		print(attemptedPostAbsFFT2D)

		p.matshow(preFFT2D, cmap="Greys_r")
		p.show()

		p.matshow(attemptedPostAbsFFT2D, cmap="Greys_r")
		p.show()

		p.matshow(postFFT2D, cmap="Greys_r")
		p.show()