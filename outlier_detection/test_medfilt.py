import numpy as np
import scipy
import matplotlib.pylab as plt
import scipy.optimize as mcf
import scipy.signal as ss

filter_size = 51
fname = 'runmed'




def movingaverage (x,y, window):
 weights = np.repeat(1.0, window)/window
 sma = np.convolve(y, weights, 'valid')
 sma_x = x[:]
 sma_y = np.concatenate((np.ones(window)*sma[0],sma))
 return(sma_x,sma_y)



sd_fake = 3.0
n_fake  = 1000

#some subset of the fake data will be the 'test outliers' the code is tasked to identify.
#the parameters should be setup so that n_outlier < n_fake and sd_outlier > sd_fake.
n_outlier = 23
sd_outlier = 5

data_y = np.random.randn(n_fake)*sd_fake
id_test = np.random.choice(np.arange(n_fake), size=n_outlier, replace=False)
data_y[id_test] = np.random.randn(n_outlier)*sd_outlier

#also have the option of including irregularly spaced time-series data by speciffying a 
#unique input array for the x-axis (code assumes regular sampling if no input)
data_x = np.arange(n_fake)



#test running median
rmed = ss.medfilt(data_y, kernel_size=filter_size)



#test running mean
model_x,rmean = movingaverage(data_x,data_y,filter_size)


plt.plot(data_y)
plt.plot(rmed)
plt.plot(rmean)
plt.show()



def func1(x,p0,p1):
 return(p0+x*p1)
 
def func2(x,p0,p1,p2):
 return(p0+p1*x+p2*x**2)
popt, pcov = mcf.curve_fit(func2, data_x, data_y)
model_y = popt[0] + data_x*popt[1] + data_x*popt[2]





