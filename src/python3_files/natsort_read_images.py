
# coding: utf-8

# In[14]:

import os
import numpy as np
import cv2
import pickle
from best_matrix import padIntegerWithZeros

def downsample_box_half(x):
     return 0.25 * (
            x[::2,::2,:] + \
            x[1::2,::2,:] + \
            x[1::2,1::2,:] + \
            x[::2,1::2,:])
    
# path: file name beginning such as "/path/to/blah/frame_test_"
# file_start: int which should be the first file read in the sequencce
# file_end: int which should be the last file read in the sequencce
# num_downsample: int which defines number of times the images should be downsampled by half.
def load_tif(path, file_start, file_end, num_downsample):
    images = []
    for i in range(file_start, file_end):
        print(i)
        img = cv2.imread(path + padIntegerWithZeros(i, 4) + ".tif", cv2.IMREAD_UNCHANGED)

        colored = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        
        colored = np.array(colored, dtype=np.float32)
        for _ in range(num_downsample):
            colored = downsample_box_half(colored)
        images.append(colored)
    return np.array(images)

imgs = load_tif("/Users/adamyedidia/walls/tsne_exp/merls_2018-11-26-153635-", \
    0, 815, 4)

print(imgs.shape)
pickle.dump(imgs, open("prafull_ball.p", "w"))




# In[ ]:



