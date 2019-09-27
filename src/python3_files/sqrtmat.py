
import numpy as np
from scipy.linalg import circulant as circ
from scipy.signal import max_len_seq as mls
from scipy.linalg import hankel
from math import sqrt, exp, pi
import matplotlib.pyplot as pl
import scipy.integrate as integrate
from scipy.linalg import sqrtm

def gaussian(i, j, sigma):
    return exp(-(i-j)**2/sigma**2)

#def gaussian(i, j):
#    print i, j

#    return np.vectorize(gaussian)(i, j)

#n = 100
#sigma = 5

#kern = np.array([[gaussian(i, j, sigma) for j in range(n)] for i in range(n)])

#kern = np.fromfunction(gaussian, (10, 10))

#pl.matshow(kern)
#pl.show()

#matsqrt = sqrtm(kern)
#print matsqrt

#s = np.random.randint(0,2,n)


n = 100
#s = list(mls(4)) + [0]

print(mls(5))

print(np.linalg.det(circ([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1,
       1, 1, 0, 1, 1, 0, 0, 0])))


#c = circ(mls(5))

#print np.linalg.det(c)

#b = np.dot(c, matsqrt)




#pl.matshow(np.real(b))
#pl.show()
