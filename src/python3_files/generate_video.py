import numpy as np
import pickle
import sys
from video_processor import convertArrayToVideo

arrName = sys.argv[1]

arr = np.array(pickle.load(open(arrName + ".p", "rb")))
#        arr = imageify(np.array(pickle.load(open(arrName + ".p", "r"))))

#        viewFrame(arr[0], adaptiveScaling=True)

print(arr.shape)

#        print arr

#        convertArrayToVideo(np.array(arr), 30000, arrName, 15, adaptiveScaling=False, differenceImage=True)
#        convertArrayToVideo(np.array(arr), 0.5, arrName, 15, adaptiveScaling=True, differenceImage=True)
convertArrayToVideo(np.array(arr), 1, arrName, 15, adaptiveScaling=True, 
    differenceImage=False, verbose=True)