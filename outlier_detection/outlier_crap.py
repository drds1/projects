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


id_out = orej.outlier_smooth(data_y,sd_check=3,fname='running median',runtype='series',filter_size = 5,max_iteration=100,diagnostic_figure='show')
