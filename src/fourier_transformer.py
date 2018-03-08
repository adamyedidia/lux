from __future__ import division
import pylab
import imageio
import numpy as np
from math import acos, pi, cos, sin
import matplotlib.pyplot as p
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pickle
from PIL import Image
from PIL import ImageFilter
import sys

imRaw = Image.open("/Users/adamyedidia/image_to_fourier.png").convert("L")
frame = np.array(imRaw.resize((1000,1000))).astype(float)


result = np.fft.fft2(frame)

print result

p.matshow(np.minimum(-np.real(result), 1e3), cmap="gray")
p.colorbar()

p.show()
