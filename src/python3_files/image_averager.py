import sys
import os
from cr2_processor import convertRawFileToArray
import pickle
from video_magnifier import viewFrame
import numpy as np

dirPath = "/Users/adamyedidia/flags_garbled/uk_garbled_small/"
filesInPath = os.listdir(dirPath)



for i, filenameShort in enumerate(filesInPath):
    print(i, "/", len(filesInPath))

    filename = dirPath + filenameShort

    thisArray = convertRawFileToArray(filename)

    if i == 0:
        arrayShape = thisArray.shape
        totalArray = np.zeros(arrayShape)

    totalArray += thisArray

totalArray /= len(filesInPath)

viewFrame(totalArray)

pickle.dump(totalArray, open(dirPath + "average.p", "w"))
