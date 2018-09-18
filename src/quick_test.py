from math import pi
import numpy as np
import matplotlib.pyplot as p

def f(x):
    return np.abs(1/(1-np.exp(1j*x)))
    
xVals = np.linspace(-pi, pi, 10000)    
    
p.plot(xVals, [f(x) for x in xVals])
p.show()
