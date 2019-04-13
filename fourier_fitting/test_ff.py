import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import sklearn.linear_model
import scipy.optimize as so
from prediction_functions.signal_decomp import *

'''
input parameters
'''
n = 100
wtrue = [3.0,17.0]
samps = [1.0,3.5]
camps = [0.0,1.4]

'''
Generate fake data
'''
t = np.arange(n)
y = np.zeros(n)
dt = t[1]-t[0]
idx = 0
for w in wtrue:
    y = y + samps[idx]*np.sin(2*np.pi/w*t) + camps[idx]*np.cos(2*np.pi/w*t)
idx = idx + 1
y = 23.*t + 0.3*t**2

'''
setup parameter grid
'''
f = np.arange(1./n,1/2/dt,1./n)
#f = np.random.choice(f,size=10,replace=False)
nf = len(f)
w = 1./f
nw = int(nf)
X = []
for iw in range(nw):
    X.append(np.sin(2*np.pi*f[iw]*t))
    X.append(np.cos(2*np.pi*f[iw]*t))
X = np.array(X).T



'''
fit model
'''
parms, cov, r2, mse, importance = constituents_fit(y, X)
pred = np.zeros(n)
for iw in range(0,2*nw,2):
    pred = pred + parms[iw]*X[:,iw] + parms[iw+1]*X[:,iw+1]

'''
compute correlation matrix
'''
sd = np.sqrt(np.diag(cov))
nparms = len(sd)
r2 = 1.*cov
for ix in range(nparms):
    for iy in range(nparms):
        r2[ix,iy]=r2[ix,iy]/sd[ix]/sd[iy]


'''
plot results
'''
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(t,y,label='true')
ax1.plot(t,pred,label='modelled')
ax1.legend()

tl = []
for iw in range(nw):
    tl.append('sin '+np.str(np.round(w[iw],1)))
    tl.append('cos '+np.str(np.round(w[iw],1)))

ax2 = fig.add_subplot(212)
a = ax2.imshow(r2)
cbar = fig.colorbar(a)
cbar.set_label('r2')
ax2.set_xlabel('p')
ax2.set_ylabel('p')
ax2.set_xticks(np.arange(nparms))
ax2.set_xticklabels(tl)
ax2.set_yticks(np.arange(nparms))
ax2.set_yticklabels(tl)
ax2.tick_params(axis='y',labelsize=6)
ax2.tick_params(axis='x', rotation=90,labelsize=6)
plt.show()