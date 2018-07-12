import pickle
import numpy as np

m = 11
n = 11

for a in range(n):
    for b in range(m):
        returnArray = []

        for c in range(n):
            returnArray.append([])

            for d in range(m):
                if a == c and b == d:
                    returnArray[-1].append([255, 255, 255])
                else:
                    returnArray[-1].append([0,0,0])

        pickle.dump(np.array(returnArray), open("impulse_" + str(a) + "_" + str(b) + \
            "_" + str(m) + "_" + str(n) + ".p", "w"))
