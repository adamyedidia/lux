
import numpy as np
import matplotlib.pyplot as p
from math import log


N = 1200
logdets = []
nlogn = []
dets = []
onePointFive = []

for n in range(1, N+1):
    randomMatrix = np.random.random((n, n))
    logdet = max(np.linalg.slogdet(randomMatrix)[1], 0)

    logdets.append(logdet)
    nlogn.append(n/2.5 * log(n) - n - 50)

    dets.append(np.linalg.det(np.identity(n) + randomMatrix/n))
    onePointFive.append(1.5)

p.plot(logdets)
p.plot(nlogn)
#p.show()
p.clf()
p.plot(dets)
p.plot(onePointFive)
p.show()
