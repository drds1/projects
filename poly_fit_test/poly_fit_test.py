from numpy.polynomial import polynomial as P
import numpy as np

x = np.linspace(-1,1,51) # x "data": [-1, -0.96, ..., 0.96, 1]
y = x**3 - x + np.random.randn(len(x))+8 # x^3 - x + N(0,1) "noise"
c, stats = P.polyfit(x,y,3,full=True)




import matplotlib.pylab as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x,y)


a = np.polyfit(x,y,3,full=False,cov=True)