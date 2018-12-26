import numpy as np
import matplotlib.pylab as plt



def sigmoid(x,parms,lo=0, hi = 1.0):
	af = parms[0] + np.sum(parms[1:]*x,axis=1)
	sigm = lo + (hi - lo)/(1. + np.exp(-af))
	return(sigm)


def likelihood(y,x,parms):
	sigm = sigmoid(x,parms)
	l = sigm**y * (1. - sigm)**((1.-y))
	return(np.prod(l))	
	
def loglikelihood(y,x,parms):
	sigm = sigmoid(x,parms)
	ll = y*np.log10(sigm) + (1. - y)*np.log10(1. - sigm)
	return(np.sum(ll))
	
	


class logreg:
	
	
	#example problem
	lo = 0.0
	hi = 1.0
	x = np.ones((100,1))
	x[:,0] = np.linspace(-10,10,100)
	y = np.ones(100)*0.5
	y[:49] = 0
	y[51:] = 1 
	parms  = np.array([1.,1.])
	
	
	def fit(self, x, y):
		self.x = x
		self.y = y
		#MORE FIT LIKILIHOOD USING NEWTON RHAPSON
		#OPTIMIZE PARAMETERS, DEFINE A PREDICT AND 
		#EVALUATE MODEL FUNCTIONS WITHIN CLASS
	
	def set_parms(self, parms):
		self.parms = parms
	def set_lo(self, lo):
		self.lo = lo
	def set_hi(self, hi):
		self.hi = hi
	
	def get_parms(self):
		print(self.parms)
	def get_lo(self):
		print(self.lo)
	def get_hi(self):
		print(self.hi)
	
 
	def plot_sigmoid(self):
 		ndim = np.shape(self.x[0,:])[0]
 		plt.clf()
 		fig = plt.figure()
 		for i in range(ndim):
 			ax1 = fig.add_subplot(ndim,1,i+1)
 			ymodel = sigmoid(self.x,self.parms,lo=self.lo,hi=self.hi)
 			ax1.plot(self.x[:,i],self.y,label='sigmoid data')
 			ax1.plot(self.x[:,i],ymodel,label='sigmoid model')


		
	
	
		
		
		

a = logreg()

a.get_parms()
	
	
	
	
 
 