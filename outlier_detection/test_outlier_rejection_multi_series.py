import numpy as np
import scipy
import matplotlib.pylab as plt
import scipy.optimize as mcf
import scipy.signal as ssig
import matplotlib.gridspec as gridspec
import outlier_rejection as outr
plt.clf()
#this time generate n_ts random time-series and introduce anomalies in one of the timeseries
#now ask the code to identify outlying points in the anomalous time series
#given the other time series as references

sd_background = 3.0
n_epoch  = 1000
n_timeseries = 100
id_outlier = 23
time_anomaly = 350
grad_anomaly = 0.1




data_y = np.reshape( np.random.randn(n_epoch * n_timeseries), (n_epoch,n_timeseries) )




#introduce a linear gradient into one timeseries to simulate non-stationarity
data_y[time_anomaly:,id_outlier] += data_y[time_anomaly,id_outlier] + np.arange(0,n_epoch-time_anomaly,1)*grad_anomaly
plt.plot(data_y[:,id_outlier])
plt.show()




#now compute the sigma-clip routine across parallel time series one time at a time

id_outliers = np.zeros((2,0))
for i in range(n_epoch):
 y_now = data_y[i,:]
 x_pass,y_pass,idx_outlier = outr.outrej(y_now,sd_check=3.5,fname='running median',filter_size = 5,max_iteration=10,diagnostic_figure='')
 
 #save the points identified as outliers in a 2d array with 0th column corresponding to
 #time series ID and 1st column to the epoch id
 n_outliers = np.shape(idx_outlier)
 id_out = np.vstack( (np.ones(n_outliers)*i,idx_outlier) )
 id_outliers = np.hstack((id_outliers,id_out))
id_outliers = id_outliers.T[:,[1,0]]


#plot the results
fig = plt.figure()
ax1 = fig.ad_subplot
gs1 = gridspec.GridSpec(4, 4)
gs1.update(left=0.1, right=0.9, bottom=0.1,top = 0.9, wspace=0.05,hspace = 0.0)
ax1 = plt.subplot(gs1[:3, :])
ax1.plot(data_y[0,:],label='Time series',color='k')
ax1.plot(data_y[0,:],label=None,color='k')
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