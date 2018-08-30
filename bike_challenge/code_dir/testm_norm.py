#it seems like draws from multivariate gaussian is an O(N^2) process

import numpy as np
from scipy.stats import multivariate_normal as ms
import time
import matplotlib.pylab as plt


nits = 500

tsave = []
ssave = [5000]#np.logspace(1,3.3,20,dtype='int')

for s in ssave:

 mean = np.zeros(s)
 cov = np.diag(np.ones(s))
 
 
 t1 = time.time()
 mn = np.random.multivariate_normal(mean,cov,size=nits)
 t0 = time.time()
 tsave.append(t0-t1)
 

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(ssave,tsave)
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.savefig('plt_test_mvtime.pdf')