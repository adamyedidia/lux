from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as p

def transform2DOccluderToMatrix(occ, extra):
    occShape = occ.shape
    print(occShape)

    matrixArray = []

    for a in range(occShape[0]/2):
        for b in range(occShape[1]/2):
            matrixArray.append([])

            for i in range(a+extra, a + occShape[0]/2+extra):
                for j in range(b, b + occShape[1]/2):
                    matrixArray[-1].append(occ[i][j])

    return np.array(matrixArray)


nOver2 = 10
occluder1 = np.array([[1]*nOver2 + [0]*nOver2] * nOver2 + [[0]*nOver2*2]*nOver2)
occluder2 = np.array([[1]*nOver2*2]*nOver2 + [[0]*nOver2 + [1]*nOver2] * nOver2)

p.matshow(occluder1)
p.show()
p.matshow(occluder2)
p.show()

tm1 = transform2DOccluderToMatrix(occluder1, 0)
tm2 = transform2DOccluderToMatrix(occluder2, 1)


print(tm1.shape)

p.matshow(tm1)
p.show()
p.matshow(tm2)
p.show()

p.matshow(np.linalg.inv(tm1))
p.show()
p.matshow(np.linalg.inv(tm2))
p.show()

#image = np.random.random((nOver2*2, nOver2*2))
#p.matshow(image)
#p.show()

#obs1 = convolve2d(occluder1, image)

#p.matshow(obs1)
#p.show()

#obs2 = convolve2d(occluder2, image)

#p.matshow(obs2)
#p.show()



#p.matshow(occluder1)
#p.show()
#p.matshow(occluder2)
#p.show()
