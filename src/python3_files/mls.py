import numpy as np
from scipy.linalg import circulant as circ
from scipy.signal import max_len_seq as mls
from video_magnifier import viewFrame, viewFrameR



#s = np.array([1,2,3])
#print

s = mls(6)[0]

print(s)

def blackAndWhiteifyFlatFrame(flatFrame):
    return np.swapaxes(np.array([flatFrame]*3),0,1)

def displayFlattenedFrame(flattenedFrame, height, magnification=1, \
    differenceImage=False, filename=None):

    frame = np.array([flattenedFrame]*height)
    viewFrame(frame, differenceImage=differenceImage, magnification=magnification,
        filename=filename)

displayFlattenedFrame(blackAndWhiteifyFlatFrame(s), 20,\
    magnification=255, differenceImage=False)


#e = np.linalg.eig(circ(s))[0]
#print e
#print np.multiply(np.conjugate(e), e)

#print np.linalg.det(circ(s))
