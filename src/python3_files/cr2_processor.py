#!/usr/bin/env python

import rawpy
from video_magnifier import viewFrame
from video_processor import batchAndDifferentiate
import pickle

process = False
process2 = True


#filename = '/Users/adamyedidia/flags_garbled/calibration/IMG_5048.CR2'
#filename = '/Users/adamyedidia/flags_garbled/texas_garbled/IMG_5051.CR2'

#raw = rawpy.imread(filename)
#bayer = raw.raw_image # with border
#bayer_visible = raw.raw_image_visible # just visible area

#print bayer

def convertRawFileToArray(filename, batchSideLength):
    with rawpy.imread(filename) as raw:
        rgb = raw.postprocess()

    arr = rgb.astype(float)
    batchedArr = batchAndDifferentiate(arr, [(batchSideLength, False), \
        (batchSideLength, False), (1, False)])

    return batchedArr

if __name__ == "__main__":


    if process:
    #    path = "/Users/adamyedidia/walls/src/pole_images/legos/back/"
    #    path = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/b_dark/"
        path = "/Users/adamyedidia/Dropbox (MIT)/shadowImaging/2d-imaging/data/" + \
            "2017-11-05/doorway/pointlight/point1.cr2"


        print("Converting...")

        arr = convertRawFileToArray(path + "all_white_2.CR2", 1)

        print("Writing...")

        pickle.dump(arr, open(path + "average.p", "w"))

    if process2:
    #    path = "/Users/adamyedidia/walls/src/pole_images/legos/back/"
    #    path = "/Users/adamyedidia/walls/src/pole_images/monitor_lines/b_dark/"
        path = "/Users/adamyedidia/Dropbox (MIT)/shadowImaging/2d-imaging/data/" + \
            "2017-11-05/doorway/pointlight/point1"


        print("Converting...")

        arr = convertRawFileToArray(path + ".cr2", 10)

        print("Writing...")

        pickle.dump(arr, open(path + ".p", "w"))






    #print convertRawFileToArray(filename)

    #rgb =
    #viewFrame(rgb)
    #viewFrame(bayer_visible)
