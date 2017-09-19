from math import tan, atan, sqrt, asin, pi

dy = 0
dx = 0
theta = pi/4
r = 1

beta = atan(dx/dy)
maxLight = r - dy*tan(theta) + dx
f = r - maxLight/2
alpha = asin(sqrt(dx*dx + dy*dy)*sin(theta-beta)/f)
gamma = pi - alpha - theta + beta
phi = pi - gamma + beta #

print phi
