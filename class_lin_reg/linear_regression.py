import numpy as np
import matplotlib.pylab as plt



#trial run through make simple linear regression code in class format
#later apply to time series forecasts


def poly(x,p):
	xlist = list(x)
	ylist = []
	n_p = len(p)
	for xnow in xlist:
		ylist.append(np.sum([p[-1-i]*xnow**i for i in range(n_p)]))
	return(np.array(ylist))

class linreg:
	
	
	#example problem
	lo = 0.0
	hi = 1.0
	x = np.ones((100,1))
	x[:,0] = np.linspace(-10,10,100)
	y = 0.5*x + 2*x**2 + 0.1
	parms  = np.array([1.,1.,1.0])
	
	
	def fit(self, x, y):
		self.x = x
		self.y = y
		print('fitting params',self.parms)
		p,cov = np.polyfit(x,y,len(self.parms)-1,cov=True)
		self.parms = p
		print('after params',self.parms)
		
	def predict(self,x):
		self.predictions = np.sum([self.parms[-1-i]*x**i for i in range(len(self.parms))])
	
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
	
 
	def plot_fit(self):
 		plt.clf()
 		fig = plt.figure()
 		ax1 = fig.add_subplot(1,1,1)
 		ymodel = poly(self.x,self.parms)
 		ax1.scatter(self.x,self.y,label='data')
 		ax1.plot(self.x,ymodel,label='model')


		
	
	
		
		
		

a = linreg()

print(a.get_parms(),'before')

x = np.linspace(-10,10,100)
y = 0.5*x + 2*x**2 + 0.1
p = np.ones(3)

a.set_parms(p)

print(a.get_parms(),'after')

a.fit(x,y)
a.plot_fit()
plt.show()
	
	
	
 
 