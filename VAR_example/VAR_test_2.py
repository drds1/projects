import numpy as np
import matplotlib.pylab as plt
import pandas

import statsmodels.api as sm

from statsmodels.tsa.api import VAR, DynamicVAR

mdata = sm.datasets.macrodata.load_pandas().data

 # prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)

quarterly = dates["year"] + "Q" + dates["quarter"]

from statsmodels.tsa.base.datetools import dates_from_str

quarterly = dates_from_str(quarterly)

mdata = mdata[['realgdp','realcons','realinv']]

mdata.index = pandas.DatetimeIndex(quarterly)

data = np.log(mdata).diff().dropna()

# make a VAR model
model = VAR(data)


results = model.fit(maxlags=15, ic='aic')
results.summary()

lag_order = results.k_ar

#produce combined 5 step forecast 
fc_comb = results.forecast(data.values[-lag_order:], 5)
results.plot_forecast(5)
plt.show()

#produce iterated 5 step forecast --> Verified these are the same 
#therefore code incorporates this step :)
fc_it = []
last_vals = data.values[-lag_order:]
nsteps = 5
fc_now = []
for i in range(nsteps):
	fc_now.append(results.forecast(last_vals, 1)[0])
	last_vals[:-1,:] = last_vals[1:,:]
	last_vals[-1,:] = fc_now[-1]
fc_now = np.array(fc_now)
