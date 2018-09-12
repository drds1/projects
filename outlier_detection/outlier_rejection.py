
# coding: utf-8

# This function aims to apply sigma-clipping to reject outlying data points in time-series data. The model iterativelty fits a smooth polynomial function (with the order as an input parameter) and rejects outliers beyond n-standard deviations of the fit (n is the second input parameter). The iterations end when no further poins are rejected or after a maximum of 100 is reached (the process should never take this long and a flag will raise if this occurs).
# 
# First import requried modiles.

# In[1]:


import numpy as np
import scipy
import matplotlib.pylab as plt
import scipy.optimize as mcf
import scipy.signal as ssig
import matplotlib.gridspec as gridspec


filter_size = 5
fname = 'running median'


# generate some fake random data to test the code. Specify the parameters of the fake data below

# In[2]:


sd_fake = 3.0
n_fake  = 1000

#some subset of the fake data will be the 'test outliers' the code is tasked to identify.
#the parameters should be setup so that n_outlier < n_fake and sd_outlier > sd_fake.
n_outlier = 23
sd_outlier = 10.0

data_y = np.random.randn(n_fake)*sd_fake
id_test = np.random.choice(np.arange(n_fake), size=n_outlier, replace=False)
data_y[id_test] = np.random.randn(n_outlier)*sd_outlier

#also have the option of including irregularly spaced time-series data by speciffying a 
#unique input array for the x-axis (code assumes regular sampling if no input)
data_x = np.arange(n_fake)


# Define the outlier rejection function. 
# 
# 
# Arguments are as follows...
# 
# INPUTS: 
# data_y_in, data_x_in --> 1D array containing the values of the time-series.
# 
# 
# 
# OPTIONAL INPUTS:
# 
# data_x_in            --> 1D arrays containing the time axis of the time-series (if no data_x_in entered, code assumes evenly spaced data).
# 
# 
# sd_check             --> The number of standard deviations from the fitted model (see fname) to consider a point outliying.
# 
# fname                --> The type of smooth model to fit to the time series prior to rejection (linear, polynomial, running mean, running median).
# 
# filter_size          --> The size of the window to use for running mean and median computation (for running mean and median functions only).
# 
# max_iteration        --> Defines the stopping point for the sigma clipping (either when no further points are rejected or when max_iteration is reached).
# 
# 
# diagnostic_figure    --> The name of an output figure showing the time-series with the outliers flagged. If blank ('') then no plot is made, else enter something like 'output_figure.pdf'
# 
# OUTPUTS:
# data_x               --> The output time-stamps (will be the indicees of accepted points if no input provided).
# 
# data_y               --> The y-axis of accepted points in the time-series.
# 
# idx_out              --> The indicees of rejected points of the input time-series (data_y_in)

# In[3]:


def outrej(data_y_in,data_x_in=[],sd_check=4,fname='running median',filter_size = 5,max_iteration = 10,
           diagnostic_figure=''):


 def func1(x,p0,p1):
  return(p0+x*p1)
 def func2(x,p0,p1,p2):
  return(p0+p1*x+p2*x**2)
 
 def movingaverage (x,y, window):
  weights = np.repeat(1.0, window)/window
  sma = np.convolve(y, weights, 'valid')
  sma_x = x[:]
  sma_y = np.concatenate((np.ones(window)*sma[0],sma))
  return(sma_x,sma_y)
 
 
 #if time values not provided, assume even sampling.
 if (type(data_x_in) == np.ndarray):
  data_x = np.array(data_x_in)
 else:
  ndat   = np.shape(data_y_in)[0]
  data_x = np.arange(ndat)
    
 data_y = np.array(data_y_in)
 out_x = []
 out_y = []
 idx_out = []
    
 #perform the iterations until max_iteration or no further rejections made   
 for iteration in range(max_iteration):
  
  #fit smooth function (linear, polynomial, running mean, running median)
  if (fname == 'linear'): 
   popt, pcov = mcf.curve_fit(func1, data_x, data_y)
   model_y = popt[0] + data_x*popt[1]
   model_x = np.array(data_x)
  elif (fname == 'quadratic'):
   popt, pcov = mcf.curve_fit(func2, data_x, data_y)
   model_y = popt[0] + popt[1]*data_x + popt[2]*data_x**2
   model_x = np.array(data_x) 
  elif (fname == 'running median'):
   model_y = ssig.medfilt(data_y, kernel_size=filter_size)
   model_x = np.array(data_x)
  elif (fname == 'running mean'):
   model_x,model_y = movingaverage(data_x,data_y,filter_size)
 
 
  #compute the residuals ( abs[data-model])
  #model_itp = np.interpolate(data_x,model_x,model_y)   
  residual = data_y - model_y
 
     
  #identify points greater than sd_outlier from the model 
  sd = np.std(residual)
  id_out = np.where(np.abs(residual) > sd_check*sd)[0]
  n_out = np.shape(id_out)[0] 
 
     
  #flag outliers and remove from them from the data arrays for the next iteration
  #for id in id_out:
  # print data_x[id],data_y[id]
 
  out_x.append(data_x[id_out])
  out_y.append(data_y[id_out])
  data_x = np.delete(data_x,id_out)
  data_y = np.delete(data_y,id_out)
 
  #save a record of rejected array indicees 
  idx_out.append(id_out)

  #exit the loop prematurely if no outliers found
  if (np.shape(id_out)[0] == 0):
   print 'no further oultiers found after iteration ',iteration,' exiting...'
   break
  else:
   print 'iteration ',iteration,'\n ',n_out,'outliers found'
    
 #flag a warning if the outlier rejection has not converged after max_iteration iterations
 if (iteration == max_iteration - 1):
  print 'warning: Outlier rejection did not converge after a maximum,', max_iteration, ' iterations re-run for more iterations or check input data for bugs'
 
    
    
 idx_out = np.concatenate(idx_out)   
 #plot a diagnostic plot if requested
 if (diagnostic_figure != ''):
  
  gs1 = gridspec.GridSpec(4, 4)
  gs1.update(left=0.1, right=0.9, bottom=0.1,top = 0.9, wspace=0.05,hspace = 0.0)
  ax1 = plt.subplot(gs1[:3, :])
  ax1.plot(data_x,data_y,label='Time series',color='k')
  ax1.plot(model_x,model_y,label='Smooth model',color='blue')
  ax1.scatter(data_x_in[idx_out],data_y_in[idx_out],marker='o',color='red',label='Outliers')
  plt.legend()
  ax1.set_xlabel('Time')
  ax1.set_ylabel('Time-series values')
  
  axres = plt.subplot(gs1[3:, :])
  axres.plot(data_x,residual)
  axres.set_xlabel('Time')
  xl = list(axres.get_xlim())
  axres.set_xlim(xl)
  ax1.set_xlim(xl)
  axres.plot(xl,[0,0],ls=':')
  axres.set_ylabel('residuals \n (data - model)') 

  if (diagnostic_figure != 'show'):
   plt.show()
  else:
   plt.savefig(diagnostic_figure)
        
 return(data_x,data_y,idx_out)



# Call the outlier rejection function defined above and test on the fake data.

# In[5]:
window = 5
x = data_x
y = data_y
weights = np.repeat(1.0, window)/window
sma = np.convolve(y, weights, 'valid')
sma_x = x[:]
sma_y = np.concatenate((np.ones(window-1)*sma[0],sma))


data_x_pass,data_y_pass,idx_outlier = outrej(data_y,data_x,sd_check=4,fname='running mean',filter_size = 5,max_iteration=10,diagnostic_figure='show')

