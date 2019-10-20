import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as p
#from import_1dify import stretchArray

#vid = pickle.load(open("smaller_movie_batched_diff_framesplit.p", "rb"), encoding="latin1")

#pickle.dump(vid[1200], open("sparse_vickie_movement.p", "wb"))

CONVERTER = False
GENERAL_CONVERTER = True
MAKE_GAUSS = False

if __name__ == "__main__":

	if CONVERTER:

		imSize = 28
		moniker = "circle"

		im = pickle.load(open(moniker+"_"+str(imSize)+"_"+\
		            str(imSize)+".p", "rb"), encoding="latin1")

		pickle.dump(im, open(moniker+"_"+str(imSize)+"_"+\
		            str(imSize)+".p", "wb"))

	if GENERAL_CONVERTER:

		imSize = 28
		moniker = "dora_slightly_downsampled"

		im = pickle.load(open(moniker+".p", "rb"), encoding="latin1")

		pickle.dump(im, open(moniker+"_python3friendly.p", "wb"))

	if MAKE_GAUSS:
		imSize = 28
		t = np.linspace(-10, 10, imSize)
		bump = np.exp(-0.1*t**2)
		bump /= np.trapz(bump) # normalize the integral to 1

		moniker = "gauss"

		# make a 2-D kernel out of it
		im = bump[:, np.newaxis] * bump[np.newaxis, :]

		p.matshow(im, cmap="Greys_r")
		p.show()

		pickle.dump(im, open(moniker+"_"+str(imSize)+"_"+\
            str(imSize)+".p", "wb"))