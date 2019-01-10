import mylcgen as mlc
import numpy as np
import myconvolve as mc
import pandas as pd
lag = 10.0
wide = 2.0

#construct mock time series and include impulse response lag function
dat = mlc.mylcgen(datfile='', p0=1.0, f0=0.1, a=-2, b=-2, tlo=0, thi=100,
	  dt=0.1, ploton=0, iseed=13662535, meannorm=-1.0, sdnorm=-1.0)
tau = np.arange(0,100,0.1)
response = np.exp(-0.5*(tau - lag)**2/wide**2)/(2*np.pi*wide**2)
echo = mc.mc3(dat[:,0],dat[:,1],tau,response)

n = np.shape(echo)[0]


from pylab import *
plot(dat[:,0],dat[:,1])
plot(dat[:,0],echo)
savefig('test_timeseries.png')

data = np.zeros((n,2))
data[:,0] = dat[:,1]
data[:,1] = echo
data = data[::10,:]

X,y = data[:,0],data[:,1]
data = pd.DataFrame(data,columns=['drive','echo'])



import linvpy as lp

#recast driving time series into N X Nlag matrix A for linvpy process
idlo,idhi = -20,20
idlag = np.arange(idlo,idhi+1,1)
lenlag = len(idlag)
ndrive = np.shape(X)[0]
A = np.tile(X,lenlag)
A = np.reshape(A,(lenlag,ndrive))
for i in range(lenlag):
 A[i,:] = np.roll(A[i,:],idlag[i])
A = A.T

plt.clf()
tau = lp.TauEstimator(loss_function = lp.Bisquare,lamb=10.0)
result = tau.estimate(A,y)
plt.plot(result[0])
plt.show()









##need to build custom regularisation function to allow extra components to be included
##without the auto correlation penalty
##see documentation at
##https://linvpy.readthedocs.io/en/latest/#linvpy.Bisquare
#
#a = np.matrix([[1, 2], [3, 4], [5, 6]])
#y = np.array([1, 2, 3])
#tiko = lp.Tikhonov()
#tiko.regularize(a, y, 2)
#
#
#
#
#
## Define your own regularization that extends lp.Regularization
#class CustomRegularization(lp.Regularization):
#    pass
#    
#    def__init__(self,skip):
#    	self.skip = skip
#    # Define your regularization function here
#    def regularize(self, a, y, lamb=0):
#        return np.ones(a.shape[1])
#
#a = np.matrix([[1, 2], [3, 4], [5, 6]])
#y = np.array([1, 2, 3])
#
## Create your custom tau estimator with custom regularization function
## Pay attention to pass the loss function as a REFERENCE (without the "()"
## after the name, and the regularization as an OBJECT, i.e. with the "()").
#custom_tau = lp.TauEstimator(regularization=CustomRegularization())
#print custom_tau.estimate(a,y)
#
#