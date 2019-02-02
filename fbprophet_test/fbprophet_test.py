#import matplotlib
#matplotlib.use("TkAgg")
#from matplotlib import pyplot as plt

from fbprophet import Prophet
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import fish_forecast
#input parms
predict = 52*3

#load the data and convert to required format
path = '/Users/david/projects/vortexa/analytics/arbitrage_flow_study/forecasting_experiments/padd1_gasoline_imports/results/'
fb_all = pd.read_csv(path+'input_data.csv')[['Date',
       'Weekly East Coast (PADD 1) Imports of Total Gasoline  (Thousand Barrels per Day)']]
fb_all.columns = ['ds','y']
fb_all['ds'] = pd.to_datetime(fb_all['ds'])

#perform a single fit for 3 years forecast (long term)
m = Prophet(yearly_seasonality = True)
fb_data = fb_all[:-predict]
m.fit(fb_data)
future_dates = m.make_future_dataframe(periods = 7*predict)
future = m.predict(future_dates)

#plot result
fig = plt.figure()
ax1 = fig.add_subplot(111)
m.plot(future,ax=ax1)
plt.legend()
plt.savefig('fbprophet.pdf')

#perform rolling fit
rolling_prediction = []
rolling_true = []
for i in range(1,predict):
	m1 = Prophet(yearly_seasonality = True)
	fb_data = fb_all[:-i]
	m1.fit(fb_data)
	future_dates = m1.make_future_dataframe(periods = i*7)
	future = m1.predict(future_dates)
	rolling_prediction.append(future['yhat'].values[-1])
	rolling_true.append(fb_all['y'].values[-i])


#evaluate mad and correct gradient %
rolling_true = np.array(rolling_true)
rolling_prediction = np.array(rolling_prediction)
correct_gradient = fish_forecast.correct_gradient(rolling_prediction,rolling_true)
pc_correct = 100.*len(correct_gradient)/(len(rolling_true)-1)
residue = rolling_prediction - rolling_true
mad = np.median(np.abs(residue))

#plot residuals
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(residue)
ax1.set_xlabel('residue')
ax1.set_ylabel('Number')
tit = 'residual distribution \n MAD='+np.str(np.int(mad))+'	correct gradient ='+np.str(np.int(pc_correct))+'%'
ax1.set_title(tit)
plt.savefig('residual_hist.pdf')

#output results
results = {
        'MAD': [mad],
        'MAD (%)':[mad/np.median(fb_data.values[:,1])*100],
        'Matching Gradients (%)':[pc_correct]
        }
results = pd.DataFrame(results,index=['Forecast stats']).T
results.to_csv('rolling_forecast_stats.csv')




