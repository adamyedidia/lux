
import numpy as np
from image_distortion_simulator import doFuncToEachChannelVec, doFuncToEachChannel
from blind_deconvolution import vectorizedDot
import matplotlib.pyplot as p

def forwardModelMaker(transferMatrix, imSpatialDimensions):
	def forwardModel(scene):
		obs = doFuncToEachChannel(lambda x: vectorizedDot(transferMatrix, x, \
			imSpatialDimensions), scene)

		return obs
	return forwardModel