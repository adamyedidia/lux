#!/usr/bin/env python

import rawpy
from video_magnifier import viewFrame
import pickle

process = False


#filename = '/Users/adamyedidia/flags_garbled/calibration/IMG_5048.CR2'
#filename = '/Users/adamyedidia/flags_garbled/texas_garbled/IMG_5051.CR2'

#raw = rawpy.imread(filename)
#bayer = raw.raw_image # with border
#bayer_visible = raw.raw_image_visible # just visible area

#print bayer

def convertRawFileToArray(filename):
    with rawpy.imread(filename) as raw:
        rgb = raw.postprocess()

    return rgb.astype(float)

if process:
    path = "/Users/adamyedidia/flags_garbled/calibration/"

    print "Converting..."

    arr = convertRawFileToArray(path + "all_white.CR2")

    print "Writing..."

    pickle.dump(arr, open(path + "average.p", "w"))





#print convertRawFileToArray(filename)

#rgb =
#viewFrame(rgb)
#viewFrame(bayer_visible)
