
import numpy as np
import matplotlib.pyplot as p

n = 100

arr = []

for i in range(n):
    s = i + min(i+1, n-i)
    arr.append([0]*(i) + [1]*(min(i+1, n-i)) + [0]*(n-s))

def f(dv):
    if dv >= 0:
        return n*(1-dv)/(1+dv)
    else:
        return n*(1+dv)/(1-dv)



p.plot(np.linspace(1, -1, 100), [f(i) for i in np.linspace(-1, 1, 100)])
p.show()
"""
nparr = np.array(arr)

p.matshow(np.zeros((n,n)))

p.matshow(np.identity(n))

p.matshow(nparr)
p.show()
p.matshow(np.linalg.inv(nparr))
p.show()

print nparr
print np.linalg.inv(nparr)
"""
