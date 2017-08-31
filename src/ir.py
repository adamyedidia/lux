from __future__ import division
from video_magnifier import turnVideoIntoListOfFrames, viewFrame
import imageio
import pickle
import numpy as np

def dVideoDT(listOfFrames):
    returnList = []
    for i in range(len(listOfFrames) - 1):
        print i, "/", len(listOfFrames)
        returnList.append(listOfFrames[i+1] - listOfFrames[i])

    return returnList

#    return [listOfFrames[i+1] - listOfFrames[i] for i in \
#        range(len(listOfFrames)-1)]

FILE_NAME = "pendulum.m4v"
vid = imageio.get_reader(FILE_NAME,  'ffmpeg')

listOfFrames = turnVideoIntoListOfFrames(vid)

#print len(listOfFrames)

differenceVideo = dVideoDT(listOfFrames)

#pickle.dump(differenceVideo, open("diff_ir_pendulum.p", "w"))

totalVariation = 0

for i, frame in enumerate(differenceVideo):
    totalVariation += np.sum(np.square(frame))

    print i, totalVariation

    if i % 100 == 0:
        print frame
        viewFrame(np.array(frame), differenceImage=True, magnification=10, \
            filename=None)
