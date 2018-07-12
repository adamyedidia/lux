import matplotlib.pyplot as pl
from math import log
import numpy as np

def hb(p):
    return -p*log(p) - (1-p)*log(1-p)

eps = 0.1
for q in np.linspace(1e-8,1e-7,100):
    pl.plot(q, hb(q)*(1-2*eps), "bo")
    pl.plot(q, hb(q*(1-eps) + (1-q)*eps) - hb(eps), "ro")

pl.show()
