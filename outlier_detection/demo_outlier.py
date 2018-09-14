
# coding: utf-8

# Demostration of the various functions of mod_sc.py
# 
# 
# First identify Contextual anomalies (Talga et al 18 Figure 1a)

# In[1]:


import numpy as np
import scipy
import matplotlib.pylab as plt
import scipy.optimize as mcf
import scipy.signal as ssig
import matplotlib.gridspec as gridspec
import outlier_rejection as orej


#generate some fake random data to test the code. Specify the parameters of the fake data below
sd_true = 1.0
n_true  = 1000
n_outlier = 20
mean_outlier = 10.0
sd_outlier = 1.0


#make the fake data
data_y = np.random.randn(n_true)*sd_true
id_test = np.random.choice(np.arange(n_true), size=n_outlier, replace=False)
data_y[id_test] = np.random.randn(n_outlier)*sd_outlier + mean_outlier
idneg = np.random.choice(np.arange(n_outlier), size=n_outlier/2, replace=False)
data_y[id_test[idneg]] = -1*data_y[id_test[idneg]]

#Call the outlier rejection function defined above and test on the fake data.
id_out = orej.outlier_smooth(data_y,sd_check=3.5,
fname='running median',filter_size = 5,max_iteration=10,diagnostic_figure='show')


# Now flag anomalous sub-sequences within a given series

# In[2]:


subsequence_size = 100
data_y = np.random.randn(n_true)*sd_true
id_test = np.arange(n_true/2,n_true/2+subsequence_size)
data_y[id_test] = np.random.randn(subsequence_size)*sd_outlier + mean_outlier
data_y_pass,data_x_pass,model_y,idx_outlier = orej.outrej(data_y,sd_check=3.5,
fname='global median',filter_size = 5,max_iteration=10,diagnostic_figure='show')


# Now try a slightly more complex model. Simulate non-stationary time series (in this case some combination of periodic signal that increases with time as a polynomial).

# In[3]:


#make the fake data
period = 200.0
amplitude = 2
t_ref = 400
amp_poly = 2
data_y = np.random.randn(n_true)*sd_true +  amplitude*np.sin(2*np.pi/period*np.arange(n_true)) + amp_poly*(np.arange(n_true)/t_ref)**2
id_test = np.random.choice(np.arange(n_true), size=n_outlier, replace=False)

idneg = np.random.choice(np.arange(n_outlier), size=n_outlier/2, replace=False)
mirror = np.ones(n_outlier)
mirror[idneg] = -1.
data_y[id_test] = amplitude*np.sin(2*np.pi/period*id_test) + amp_poly*(id_test/t_ref)**2 + mean_outlier*mirror




# In[4]:


id_out = orej.outlier_smooth(data_y,sd_check=5,fname='running median',runtype='series',filter_size = 5,max_iteration=10,diagnostic_figure='show')


# The 'outlier_smooth' fits a smooth function to identify outliers inconsistent with the evolving time series. A standard sigma clip using just the mean and standard deviation would fail here as the distribution is now multimodal and non-stationary. 
# 
# The examples above are all looking for outliers from within a single time-series. Now introduce a set of multiple time series data with one entire anomalous time series. The objective is now to identify anomalies between multiple time series rather than within a single time series (Figure 2c from Talagala et al 2018).

# In[7]:


#make the fake data
period = 200.0
amplitude = 2
t_ref = 200
amp_poly = 2

sd_background = 3.0
n_epoch  = 1000
n_timeseries = 100
id_outlier = 23
time_anomaly = 35
grad_anomaly = 0.1
diagnostic_figure = 'show'
data_y = np.reshape( np.random.randn(n_epoch * n_timeseries), (n_epoch,n_timeseries) )

data_y[:,time_anomaly] = np.random.randn(n_epoch)*sd_background +  amplitude*np.sin(2*np.pi/period*np.arange(n_epoch)) + amp_poly*(np.arange(n_true)/t_ref)**2
id_test = np.random.choice(np.arange(n_epoch), size=n_outlier, replace=False)

idneg = np.random.choice(np.arange(n_outlier), size=n_outlier/2, replace=False)
mirror = np.ones(n_outlier)
mirror[idneg] = -1.
data_y[id_test,time_anomaly] = amplitude*np.sin(2*np.pi/period*id_test) + amp_poly*(id_test/t_ref)**2 + mean_outlier*mirror


# In[8]:


id_out = orej.outlier_smooth(data_y,sd_check=5,fname='running median',runtype='parallel',filter_size = 5,max_iteration=10,diagnostic_figure='show')


# In the above figure we have the same increasing sinusoid as with the previous example, but we also have 99 well behaved stationary time series that oscilate around zero. 'outlier_smooth' now flags the entire series as 'bad' as the iterative-smooth-model fitting now takes place between identical epochs across the 100 time series rather than within each individual time series.
