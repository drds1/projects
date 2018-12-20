import mylcgen as mlc
import numpy as np
import myconvolve as mc
import pandas as pd
lag = 10.0
wide = 2.0

#construct mock time series and include impulse response lag function
dat = mlc.mylcgen(datfile='', p0=1.0, f0=0.1, a=-2, b=-2, tlo=0, thi=100,
	  dt=0.1, ploton=0, iseed=13662535, meannorm=-1.0, sdnorm=-1.0)
tau = np.arange(0,100,0.1)
response = np.exp(-0.5*(tau - lag)**2/wide**2)/(2*np.pi*wide**2)
echo = mc.mc3(dat[:,0],dat[:,1],tau,response)

n = np.shape(echo)[0]


from pylab import *
plot(dat[:,0],dat[:,1])
plot(dat[:,0],echo)
savefig('test_timeseries.png')

data = np.zeros((n,2))
data[:,0] = dat[:,1]
data[:,1] = echo
data = data[::10,:]
data = pd.DataFrame(data,columns=['drive','echo'])

import pandas
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
#mdata = sm.datasets.macrodata.load_pandas().data
## prepare the dates index
#dates = mdata[['year', 'quarter']].astype(int).astype(str)
#quarterly = dates["year"] + "Q" + dates["quarter"]
#from statsmodels.tsa.base.datetools import dates_from_str
#quarterly = dates_from_str(quarterly)
#mdata = mdata[['realgdp','realcons','realinv']]
#mdata.index = pandas.DatetimeIndex(quarterly)
#data = np.log(mdata).diff().dropna()
#



model = VAR(data)

results = model.fit()
#plot forecast
results.plot_forecast(10)
plt.savefig('VAR_forecast.png')





#plot the response function
irf = results.irf(1000)
irf.plot(orth=False)
plt.savefig('VAR_responses.png')


