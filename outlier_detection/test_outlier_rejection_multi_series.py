import numpy as np
import scipy
import matplotlib.pylab as plt
import scipy.optimize as mcf
import scipy.signal as ssig
import matplotlib.gridspec as gridspec
import outlier_rejection as outr
plt.clf()

#now define the paralel outlier rejection sigma clip
def outrej_paralel(data_y,sd_check=5,fname='running median',filter_size = 5,max_iteration=10,diagnostic_figure=''):
 


 n_epoch,n_timeseries = np.shape(data_y)

 #now compute the sigma-clip routine across parallel time series one time at a time
 id_outliers = np.zeros((2,0),dtype='int')
 for i in range(n_epoch):
  y_now = data_y[i,:]
  y_pass,x_pass,model,idx_outlier = outr.outrej(y_now,sd_check=sd_check,fname=fname,
  filter_size = filter_size,max_iteration=max_iteration,diagnostic_figure='')
  
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
  print id_ts
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
  
  
  
#this time generate n_ts random time-series and introduce anomalies in one of the timeseries
#now ask the code to identify outlying points in the anomalous time series
#given the other time series as references

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


#test the paralel outlier rejection routine here
op = outrej_paralel(data_y,sd_check=5,fname='running median',filter_size = 5,max_iteration=10,diagnostic_figure='show')

