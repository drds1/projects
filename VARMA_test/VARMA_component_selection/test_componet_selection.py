# test partial autocorrelation to set the AR parameter rather than grid search
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pylab as plt
import numpy as np
import statsmodels.tsa.arima_process as sm

arparams = np.array([.75, -.25,0.5])
maparams = np.array([.65, .35])
ar = np.r_[1, -arparams]  # add zero-lag and negate
ma = np.r_[1, maparams]  # add zero-lag
y = sm.arma_generate_sample(ar, [0.01], 250)
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(y)
ax1.set_ylabel('time series')
ax1 = fig.add_subplot(312)
ax1.plot(pacf(y))
ax1.set_ylabel('PACF')
ax1 = fig.add_subplot(313)
ax1.plot(acf(y))
ax1.set_ylabel('ACF')
plt.show()