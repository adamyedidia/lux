import os
from best_matrix import padIntegerWithZeros

path = "dots_obs/"
oldName = "dots_scene_movie_coloravg"
newName = "frame"

maxNum = 887

for i in range(maxNum):
	os.system("mv " + path + oldName + "_" + padIntegerWithZeros(i, 3) + ".png " + \
		path + newName + "_" + padIntegerWithZeros(i, 3) + ".png")