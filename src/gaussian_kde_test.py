from scipy.stats import gaussian_kde
import jenkspy
import numpy as np
import matplotlib.pyplot as p

a = np.concatenate([np.random.normal(0, 1, 1000), np.random.normal(5,1,100)])
print a

breaks = jenkspy.jenks_breaks(a, nb_class=2)

p.hist(a,bins=30)
p.axvline(x=breaks[1], color="k")
p.show()


print breaks
