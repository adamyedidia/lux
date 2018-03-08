import numpy as np
import matplotlib.pyplot as p

n = 100
k = 10
ones = np.array([1]*(n-k))
above = np.diag(-ones, k=k)
below = np.diag(ones, k=-k)
#below = np.diag(np.array([1]*n), k=0)
together = above + below
p.matshow(together)
p.show()
inv = np.linalg.inv(together)
p.matshow(inv)
p.show()
