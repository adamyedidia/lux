from scipy.integrate import quad
import numpy as np

def integrand(x, xt, yt, ys):
    firstTrip = 1./((x-xt)**2 + yt**2)
    secondTrip = 1./(x**2 + ys**2)
    
    return firstTrip*secondTrip


    
xt = 5
yt = 0.1
ys = 5

print(quad(integrand, -np.inf, np.inf, args=(xt,yt,ys)))