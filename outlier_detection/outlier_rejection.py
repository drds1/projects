
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
from statsmodels import robust


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

# In[2]:


def outrej(data_y_in,data_x_in=[],sd_check=4,fname='running median',filter_size = 5,max_iteration = 10,
           diagnostic_figure='',comments=1,spread_function = 'rms'):


 def func1(x,p0,p1):
  return(p0+x*p1)
 def func2(x,p0,p1,p2):
  return(p0+p1*x+p2*x**2)
 
 def movingaverage (x,y, window):
  weights = np.repeat(1.0, window)/window
  sma = np.convolve(y, weights, 'valid')
  sma_x = x[:]
  sma_y = np.concatenate((np.ones(window-1)*sma[0],sma))
  return(sma_x,sma_y)
 
 
 #if time values not provided, assume even sampling.
 if (type(data_x_in) == np.ndarray):
  data_x = np.array(data_x_in)
 else:
  ndat   = np.shape(data_y_in)[0]
  data_x = np.arange(ndat)
 data_x_in_plot = np.array(data_x)   

 data_y = np.array(data_y_in)
 out_x = []
 out_y = []
 idx_out = []
    
 #perform the iterations until max_iteration or no further rejections made   
 for iteration in range(max_iteration):
  
  #fit smooth function (linear, polynomial, running mean, running median, global median or global mean)
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
  elif (fname == 'global median'):
   a =  np.median(data_y)
   ndy = np.shape(data_y)[0]
   model_y = a*np.ones(ndy)
   model_x = np.array(data_x)
  elif (fname == 'running mean'):
   model_x,model_y = movingaverage(data_x,data_y,filter_size)
  elif (fname == 'global mean'):
   a =  np.mean(data_y)
   ndy = np.shape(data_y)[0]
   model_y = a*np.ones(ndy)
   model_x = np.array(data_x)
 
  #compute the residuals ( abs[data-model])
  #model_itp = np.interpolate(data_x,model_x,model_y)   
  residual = data_y - model_y
 
     
  #identify points greater than sd_outlier from the model 
  if (spread_function == 'rms'):
   sd = np.std(residual)
  elif (spread_function == 'mad'):
   sd = robust.mad(residual)
  else:
   raise ValueError('Please ensure that the "spread_function" argument is set to "rms" or "mad"')
  
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
   if (comments == 1):
    print 'no further oultiers found after iteration ',iteration,' exiting...'
   break
  else:
   if (comments == 1):
    print 'iteration ',iteration,'\n ',n_out,'outliers found'
    
 #flag a warning if the outlier rejection has not converged after max_iteration iterations
 if ((iteration == max_iteration - 1) and (comments == 1)):
  print 'warning: Outlier rejection did not converge after a maximum,', max_iteration, ' iterations re-run for more iterations or check input data for bugs'
 
    
    
 idx_out = np.array(np.concatenate(idx_out) ,dtype='int')  
 #plot a diagnostic plot if requested
 if (diagnostic_figure != ''):
  
  gs1 = gridspec.GridSpec(4, 5)
  gs1.update(left=0.1, right=0.9, bottom=0.1,top = 0.9, wspace=0.05,hspace = 0.0)
  ax1 = plt.subplot(gs1[:3, :4])
  ax1.plot(data_x,data_y,label='Time series',color='k')
  ax1.plot(model_x,model_y,label='Smooth model',color='blue')
  ax1.scatter(data_x_in_plot[idx_out],data_y_in[idx_out],marker='o',color='red',label='Outliers')
  plt.legend()
  ax1.set_xlabel('Time')
  ax1.set_ylabel('Time-series values')
  
  axres = plt.subplot(gs1[3:, :4])
  axres.plot(model_x,residual)
  axres.set_xlabel('Time')
  xl = list(axres.get_xlim())
  axres.set_xlim(xl)
  ax1.set_xlim(xl)
  axres.plot(xl,[0,0],ls=':')
  axres.set_ylabel('residuals \n (data - model)') 

  axhist = plt.subplot(gs1[:3, 4])
  axhist.hist(data_y,orientation='horizontal',color='k',histtype='step')
  axhist.set_xticklabels([])
  axhist.set_yticklabels([])
  xy = list(ax1.get_ylim())
  axhist.set_ylim(xy)

  if (diagnostic_figure == 'show'):
   plt.show()
  else:
   plt.savefig(diagnostic_figure)
  plt.clf()
    
 return(data_y,data_x,model_y,idx_out)



# Set 'demo_single_timeseries = 1' for a demonstration

# In[3]:


demo_single_timeseries = 0


# In[4]:


if (demo_single_timeseries == 1):
 #generate some fake random data to test the code. Specify the parameters of the fake data below
 filter_size = 5
 fname = 'running median'
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
 
 #Call the outlier rejection function defined above and test on the fake data.
 data_y_pass,data_x_pass,model_y,idx_outlier = outrej(data_y,data_x,sd_check=3.5,
 fname='running median',filter_size = 5,max_iteration=10,diagnostic_figure='show')


# The function above is appropriate for identifying outliers from a single time series. With a few adaptions the function can allow for multiple time-series input.
# 
# The function below is apporpriate for cross-checking simultaneous epochs in adjacent time-series data using the sigma-clip routine in parallel epoch-by-epoch. This is useful 

# In[5]:


#now define the parallel outlier rejection sigma clip
def outrej_parallel(data_y,sd_check=5,fname='running median',filter_size = 5,max_iteration=10,diagnostic_figure='',
                    spread_function = 'rms'):
 


 n_epoch,n_timeseries = np.shape(data_y)

 #now compute the sigma-clip routine across parallel time series one time at a time
 id_outliers = np.zeros((2,0),dtype='int')
 for i in range(n_epoch):
  y_now = data_y[i,:]
  y_pass,x_pass,model,idx_outlier = outrej(y_now,sd_check=sd_check,fname=fname,
  filter_size = filter_size,max_iteration=max_iteration,diagnostic_figure='',
  comments=0,spread_function = spread_function)
  
  #save the points identified as outliers in a 2d array with 0th column corresponding to
  #time series ID and 1st column to the epoch id
  n_outliers = np.shape(idx_outlier)
  id_out = np.vstack( (np.ones(n_outliers,dtype='int')*i,idx_outlier) )
  id_outliers = np.hstack((id_outliers,id_out))
 id_outliers = id_outliers.T[:,[1,0]]
 
 
 #plot the results
 gs1 = gridspec.GridSpec(4, 4)
 gs1.update(left=0.1, right=0.9, bottom=0.1,top = 0.9, wspace=0.05,hspace = 0.0)
 ax1 = plt.subplot(gs1[:, :])
 for i in range(n_timeseries):
  if (i == 0):
   labts = 'Time series'
   labo  = 'Outliers'
  else:
   labts = None
   labo  = None
  ax1.plot(data_y[:,i],label=labts,color='k')
  id_ts = np.where(id_outliers[:,0] == i)[0]
  ax1.scatter(id_outliers[id_ts,1],data_y[id_outliers[id_ts,1],id_outliers[id_ts,0]],color='r',label=labo)
 plt.legend()
 ax1.set_title(np.str(n_timeseries)+' timeseries, ' + np.str(n_epoch) + ' epochs per series')
 ax1.set_xlabel('Time')
 ax1.set_ylabel('Time-series values')
 
 if (diagnostic_figure == 'show'):
  plt.show()
 else:
  plt.savefig(diagnostic_figure)
  
  
  
 return(id_outliers)


# The demonstration below generates 'n_timeseries' sets of timeseries data, each with 'n_epoch' points. Anomalous variability is introduced in 'id_outlier' timeseries to simulate rogue outlying points. And the 'outrej_parallel' routine is asked to identify these.
# 
# Set 'demo_multi_timeseries = 1' for a demonstration of the parallel sigma clip routine

# In[6]:


demo_multi_timeseries = 0


# In[7]:


if (demo_multi_timeseries == 1):
 sd_background = 3.0
 n_epoch  = 1000
 n_timeseries = 100
 id_outlier = 23
 time_anomaly = 350
 grad_anomaly = 0.1
 diagnostic_figure = 'show'
 
 
 
 data_y = np.reshape( np.random.randn(n_epoch * n_timeseries), (n_epoch,n_timeseries) )
 
 
 
 
 #introduce a linear gradient into one timeseries to simulate non-stationarity
 data_y[time_anomaly:,id_outlier] += data_y[time_anomaly,id_outlier] + np.arange(0,n_epoch-time_anomaly,1)*grad_anomaly
 
 
 #test the parallel outlier rejection routine here
 op = outrej_parallel(data_y,sd_check=5,fname='running median',filter_size = 5,max_iteration=10,
                      diagnostic_figure='show',spread_function = 'rms')


# One final function to perform the outrej formulation in series rather than parallel. This is the same as the first function 'outrej' but accepts simultaneous multi light curve input.

# In[8]:


#now define the parallel outlier rejection sigma clip
def outrej_series(data_y,sd_check=5,fname='running median',filter_size = 5,max_iteration=10,diagnostic_figure='',
                 spread_function = 'rms'):
 


 n_epoch,n_timeseries = np.shape(data_y)

 #now compute the sigma-clip routine across each time series one time at a time
 id_outliers = np.zeros((2,0),dtype='int')
 modelsave = []
 for i in range(n_timeseries):
  y_now = data_y[:,i]
  y_pass,x_pass,model,idx_outlier = outrej(y_now,sd_check=sd_check,fname=fname,
  filter_size = filter_size,max_iteration=max_iteration,diagnostic_figure='',
  comments=0,spread_function = spread_function)
  
  #save the smooth model
  modelsave.append(model)
  #save the points identified as outliers in a 2d array with 0th column corresponding to
  #time series ID and 1st column to the epoch id
  n_outliers = np.shape(idx_outlier)
  id_out = np.vstack( (np.ones(n_outliers,dtype='int')*i,idx_outlier) )
  id_outliers = np.hstack((id_outliers,id_out))
 id_outliers = id_outliers.T[:,[1,0]]
 
 
 #plot the results
 gs1 = gridspec.GridSpec(4, 4)
 gs1.update(left=0.1, right=0.9, bottom=0.1,top = 0.9, wspace=0.05,hspace = 0.0)
 ax1 = plt.subplot(gs1[:, :])
 for i in range(n_timeseries):
  if (i == 0):
   labts = 'Time series'
   labo  = 'Outliers'
   labmod = 'Smooth model'
  else:
   labts = None
   labo  = None
   labmod = None
  ax1.plot(data_y[:,i],label=labts,color='k')
  ax1.plot(modelsave[i],color='b',label=labmod)
  id_ts = np.where(id_outliers[:,0] == i)[0]
  ax1.scatter(id_outliers[id_ts,1],data_y[id_outliers[id_ts,1],id_outliers[id_ts,0]],color='r',label=labo)
 plt.legend()
 ax1.set_title(np.str(n_timeseries)+' timeseries, ' + np.str(n_epoch) + ' epochs per series')
 ax1.set_xlabel('Time')
 ax1.set_ylabel('Time-series values')
 
 if (diagnostic_figure == 'show'):
  plt.show()
 else:
  plt.savefig(diagnostic_figure)
  
  
  
 return(id_outliers)


# Define a function to use EVD and feature space representation of multivariate timeseries to identify outliers.

# In[9]:


import kdestats

#rolling mean function
def movingaverage(x,y, window):
  weights = np.repeat(1.0, window)/window
  sma = np.convolve(y, weights, 'valid')
  sma_x = x[:]
  sma_y = np.concatenate((np.ones(window-1)*sma[0],sma))
  return(sma_x,sma_y)

#calculate each of the features
def features(data_y,window = 10,plots='show'):
    nfeatures = 9
    
    nepoch,nts = np.shape(data_y)
    features = np.zeros((nts,nfeatures))
    
    #f1 mean
    f1 = np.mean(data_y,axis=0)
    
    
    #f2 variance
    f2 = np.var(data_y,axis=0)
    
    
    #f3 lumpiness devide into chunks defined by window size and calculate the variance of the variances
    nwindow = np.int(np.ceil(1.*nepoch/window))
    lump = np.zeros((0,nts))
    for i in range(nwindow):
     ilo = i*window
     ihi = min(ilo+window,nepoch-1)
     var = np.var(data_y[ilo:ihi,:],axis=0)
     lump = np.vstack((lump,var))
    f3 = np.var(lump,axis=0)
    
    #f4 A function to calculate Level shift using rolling window The 'level shift' is defined as the maximum difference in mean
    # between consecutive blocks of 10 observations measure6 - Level shift using rolling window
    f4 = []
    for i in range(nts):
     runmean = movingaverage(np.arange(nepoch),data_y[:,i], window)[1]
     d_runmean = runmean[1:]-runmean[:-1]
     f4.append(np.max(d_runmean))
    f4 = np.array(f4)

    #f5 calculate the rolling variance
    import pandas as pd 
    f5 = []
    for i in range(nts):
     s = pd.Series(data_y[:,i])
     run_var = np.array(s.rolling(window).var())
     d_runvar = run_var[1:]-run_var[:-1]
     f5.append(np.nanmax(d_runvar))
    f5 = np.array(f5)
    
    
    #f6 fano factor (burstiness of time series - var/mean)
    f6 = f2/f1
    
    #f7, f8 maximum and minimum of time series
    f7 = np.nanmax(data_y,axis=0)
    f8 = np.nanmin(data_y,axis=0)
    
    #f9 high to low mu ratio of means of the data that are above and below the true mean
    f9 = []
    for i in range(nts):
        mean = f1[i]
        idup = np.where(data_y[:,i]>mean)[0]
        meanhi = np.mean(data_y[idup,i])
        idlo = np.where(data_y[:,i]<mean)[0]
        meanlo = np.mean(data_y[idlo,i])
        f9.append(meanhi/meanlo)    
    f9 = np.array(f9)
    
    features[:,0]=f1
    features[:,1]=f2
    features[:,2]=f3
    features[:,3]=f4
    features[:,4]=f5
    features[:,5]=f6
    features[:,6]=f7
    features[:,7]=f8
    features[:,8]=f9
    
    labels = ['mean','variance','lumpiness','level shift mean','level shift variance',
              'burstiness','maximum','minimum','hightolowmu']
    
    
    #make diagnostic plot
    if (plots != ''):
     gs1 = gridspec.GridSpec(nfeatures, 1)
     gs1.update(left=0.1, right=0.9, bottom=0.1,top = 0.9, wspace=0.05,hspace = 0.0)
     for i in range(nfeatures):
      ax1 = plt.subplot(gs1[i, 0])
      ax1.plot(features[:,i],label=labels[i])
      #ax1.set_ylabel(labels[i])
      if (i == nfeatures - 1):
       ax1.set_xlabel('Time series ID')
      ax1.text(1.1,0.5,labels[i],ha='left',transform=ax1.transAxes)
      if (plots == 'show'):
       plt.show()
      else:
       plt.savefig(plots)
     
    return(features,labels)


#define the evd function. 2 modes, if training no need to specify pca_inp, enter training data. 
#The feature space will be calculated on the training data and PCA used to transform this
#INPUT: data_y 2D array nepochs x ntime series
#       clevel The desired confidence level to consider new points outliers
#OUTPUT: xkde,ykde = np.array(nres) The x and y values of the kernal density estimation
#        zkde = np.array((nres,nres)) The kde estimated from the training data

#If testing new data, must also input the pca_inp (output of sklearn pca routine).
#new data will be transformed into the PCA space and their kde's estimated
#INPUT: data_y 2D array of the TEST data
#       xkde,ykde = np.array(nres) The x and y values of the kernal density estimation
#        zkde = np.array((nres,nres)) The kde estimated from the training data
#       clevel the confidence level to consider new points outliers
from sklearn.decomposition import PCA
def evd(data_test,data_train,clevel=0.9973,plots='show'):
 

    
 op_test = features(data_test,window=10,plots=plots)
 op_train = features(data_train,window=10,plots=plots)
 #normalise the feature matrix
 f_train,flab_train = op_train
 f_test,flab_test = op_test
 
 nts_train,nfeatures = np.shape(f_train)
 nts_test,nfeatures = np.shape(f_test)
 fop_train = np.array(f_train)
 fop_test = np.array(f_test)
 for i in range(nfeatures):
     fop_train[:,i] = (f_train[:,i] - np.mean(f_train[:,i]))/np.std(f_train[:,i])  
     fop_test[:,i] = (f_test[:,i] - np.mean(f_test[:,i]))/np.std(f_test[:,i])  
 
 #perform PCA
 pca = PCA(n_components=2)
 pcafit = pca.fit(fop_train)
 fop_pca_train = pcafit.fit_transform(fop_train)
 fop_pca_test = pcafit.fit_transform(fop_test)



 
 #fit 2d KDE to data
 from scipy import stats
 xmin,xmax = min(np.min(fop_pca_train[:,0]),np.min(fop_pca_test[:,0])),max(np.max(fop_pca_train[:,0]),np.max(fop_pca_test[:,0]))
 ymin,ymax = min(np.min(fop_pca_train[:,1]),np.min(fop_pca_test[:,1])),max(np.max(fop_pca_train[:,1]),np.max(fop_pca_test[:,1]))

 X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
 positions = np.vstack([X.ravel(), Y.ravel()])

 xmod = X[:,0]
 ymod = Y[0,:]
 nxmod = np.shape(xmod)[0]

 values = fop_pca_train.T 
 kernel = stats.gaussian_kde(values)
 Z = np.reshape(kernel(positions).T, X.shape) 
 clevels = kdestats.confmap(Z.T, [clevel])
 
 #make a scatter plot
 if (plots != ''):
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.scatter(fop_pca_train[:,0],fop_pca_train[:,1])
  ax1.set_xlabel('PCA 1')
  ax1.set_ylabel('PCA 2')
  ax1.set_title('PCA Eigen Vector Plot (train mode)')
  ax1.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
  if (plots == 'show'):
   plt.show()
  else:
   plt.savefig(plots)
 
    
 ntrain = np.shape(fop_pca_test)[0]
 idx_x = np.array(np.interp(fop_pca_test[:,0],xmod,np.arange(nxmod)),dtype='int')
 idx_y = np.array(np.interp(fop_pca_test[:,1],ymod,np.arange(nxmod)),dtype='int') 
 zdat = Z[idx_x,idx_y]
 for i in range(len(clevels)):
  idgood = np.where(zdat > clevels[i])[0]
  idbad = np.where(zdat < clevels[i])[0]
  
  #make plot of contour levels showing good and bad time series
  if (plots != ''):
   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   ax1.scatter(fop_pca_train[:,0],fop_pca_train[:,1],label='training points')
   ax1.set_xlabel('PCA 1')
   ax1.set_ylabel('PCA 2')
   ax1.set_title('PCA Eigen Vector Plot (test mode)')
   ax1.contour(X,Y, Z, clevels)
   ax1.scatter(fop_pca_test[idbad,0],fop_pca_test[idbad,1],label='outliers test data',color='r')
   ax1.scatter(fop_pca_test[idgood,0],fop_pca_test[idgood,1],label='good points test data',color='black')
   ax1.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
   plt.legend()
   if (plots == 'show'):
    plt.show()
   else:
    plt.savefig(plots)
    
  #plot the time series in an imshow flagging up outlying series
  if (plots != ''):
   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   dis = data_test.T
   dis[idbad,:] = np.nan
   col = ax1.imshow(dis,aspect='auto')
   plt.colorbar(col,label='Time series value')
   ax1.set_xlabel('Time')
   ax1.set_ylabel('Time series ID')
   if (plots == 'show'):
    plt.show()
   else:
    plt.savefig(plots)

  return(zdat,Z,idx_x,idx_y,idbad,fop_pca_train,fop_pca_test)


# In[10]:


demo_evd = 0
if (demo_evd == 1):
 #introduce some time-varying noise into the model
 sd_ts = 4.0
 sd_epoch = 50.0
 amp_ts_max = 100
 amp_epoch = 100.0
 mean_ts = 53.0
 mean_epoch = 600.0
 
 
 
 nts = 100
 nepoch = 1000
 dat = np.random.randn(nepoch*nts).reshape(nepoch,nts)
 dat_train = np.array(dat)
 for i in range(nts):
  amp_ts   = amp_ts_max * np.exp(-0.5*((i*1. - mean_ts)/sd_ts)**2)
  dat[:,i] = dat[:,i] + amp_ts* np.exp(-0.5*((np.arange(nepoch) - mean_epoch)/sd_epoch)**2) 
  
  
 plt.clf()
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 a = ax1.imshow(dat.T,aspect = 'auto')
 plt.colorbar(a,label='Time series value')
 ax1.set_xlabel('Time')
 ax1.set_ylabel('Time series ID')
 plt.show()
 
 zdat,Z,idx_x,idx_y,idbad,fop_pca_train,fop_pca_test = evd(dat,dat_train,clevel=0.9973)
 
 




# \section{Non-stationarity}
# It is relatively easy to update this model to allow for non-stationarity. We just define a window of length 'w', input the most recent w epochs of known normal behaviour (a training data set) and the most recent w epochs. Using KDE, if > 0.5 the new observsations lie outside the confidence region of the training data, we update the KDE contours as the new 'normal behaviour'.

# In[ ]:


def evd_nonstat(train_new,test_new,clevel=0.9973,frac_check=0.5):
    
 
 zdat,Z,idx_x,idx_y,idbad,fop_pca_train,fop_pca_test =  evd(test_new,train_new,clevel=clevel,plots='')
 
 nwindow,nts = np.shape(test_new)
 frac = 1.*np.shape(idbad)[0]/nwindow
 if (frac > frac_check):
  print 'new "normal" behaviour detected. All good'
  idbad_out = []
  newnorm = 1
 else:
  idbad_out = idx_y
  newnorm = 0
 
 return(idbad_out,newnorm)

#perform the above routine over a whole time series retiurning the indicees of bad combinations
#of time series and epoch id_out[epoch,ts]
def evd_evolve(train,test,window=20,clevel=0.9973):
 nepoch,nts = np.shape(test)
 ilo = 0
 train_in = train#train[ilo:window,:]
 id_out = np.zeros((0,2))
 for i in range(nepoch-window):
  test_in  = test[i:i+window,:]
  idbad_out,newnorm = evd_nonstat(train_in,test_in,clevel=0.9973,frac_check=0.5) 
  
  if (newnorm == 1): 
   train_in = test_in

  else:
   nbad = np.shape(idbad_out)[0]
   idnow = np.ones(nbad)*i
   idn = np.zeros((nbad,2))
   idn[:,0]=idnow
   idn[:,1]=np.array(idbad_out)
   id_out = np.vstack((id_out,idn))
   
  
 return(id_out)
    
demo_nonstat = 1
if (demo_nonstat == 1):
 #introduce some time-varying noise into the model
 sd_ts = 4.0
 sd_epoch = 50.0
 amp_ts_max = 100
 amp_epoch = 100.0
 mean_ts = 53.0
 mean_epoch = 600.0
 
 
 
 nts = 100
 nepoch = 1000
 dat = np.random.randn(nepoch*nts).reshape(nepoch,nts)
 dat_train = np.array(dat)
 for i in range(nts):
  amp_ts   = amp_ts_max * np.exp(-0.5*((i*1. - mean_ts)/sd_ts)**2)
  for i2 in range(nepoch):
   if (i2 > mean_epoch):
    dat[i2,i] = dat[i2,i] + amp_ts
   else:
    dat[i2,i] = dat[i2,i] + amp_ts*np.exp(-0.5*((i2 - mean_epoch)/sd_epoch)**2) 
  
 plt.clf()
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 a = ax1.imshow(dat.T,aspect = 'auto')
 plt.colorbar(a,label='Time series value')
 ax1.set_xlabel('Time')
 ax1.set_ylabel('Time series ID')
 plt.show()





 #now test the function
 id = evd_evolve(dat_train,dat,window=20,clevel=0.9973)



# Now define a function to tie everything together

# In[ ]:


#INPUTS
#data_y enter a 1D array for single-time series anomaly detection to identify points or clusters of points
#from within a single time series, can enter a 2D array to batch run lots of time series in a single function call

#data_y enter a 2D array and set runtype 'parallel' to compute outliers at each epoch by comparing 
#with adjacent time series at the same epoch

#data_y enter a list with 2 elements, the first is a 2D array of training data n_epochs x n_timeseries, the 2nd is a
#2D array of test data. This setting will use the Extreme value distribution method of Talagala et al 2018
#to identify entire time series that exhibit anaomalies. Useful for corelated anomalies that arrise slowly
#and fool other outlier-detection methods.

#spread_function: for series modes, outliers are identified that are sd_check*spread_function from the model
# function specified in fname


def outlier_smooth(data_y,sd_check=5,fname='running median',runtype='series',filter_size = 5,
                   max_iteration=10,diagnostic_figure='',spread_function = 'rms'):

 if (runtype=='parallel'):
  if (type(data_y)==np.ndarray):
   if (len(np.shape(data_y)) !=2):
    raise ValueError('For parallel outlier detection,     data_y must be a 2D numpy array with epochs as the first axis and     different light curves as the second')
  id_outliers = outrej_parallel(data_y,sd_check=sd_check,fname=fname,filter_size = filter_size,
                                max_iteration=max_iteration,diagnostic_figure=diagnostic_figure,
                               spread_function = spread_function)
 
 #if in parallel and want to use the EVD analysis method from Talgala et al 2018, enter a list for data_y 
 #containing the test and training data. id_outliers will now return a 1d array corresponding to the 
 #outlying time series
 elif (type(data_y)==list): 
  zdat,Z,idx_x,idx_y,idbad,fop_pca_train,fop_pca_test = evd(data_y[1],data_y[0],clevel=0.9973)
  id_outliers = idbad
    
 elif (runtype == 'series'):
  if (len(np.shape(data_y)) ==1):
    a,b,c,id_outliers = outrej(data_y,sd_check=sd_check,fname=fname,
                               filter_size = filter_size,max_iteration = max_iteration,
                               diagnostic_figure=diagnostic_figure,comments=0,spread_function = spread_function)
  else:
   id_outliers = outrej_series(data_y,sd_check=sd_check,fname=fname,filter_size = filter_size,
                               max_iteration=max_iteration,diagnostic_figure=diagnostic_figure,
                              spread_function = spread_function) 

 else:
  raise ValueError('Please ensure that the "runtype" argument is specified as either parallel or series and \  that the input data is in the form of either \n   1) A 1D numpy array for a single timeseries with epochs along the axis. or \n   2) A 2D numpy array for a data cube of multiple time series with epochs as the first axis and      different time series as the second.')
 
 return(id_outliers)

