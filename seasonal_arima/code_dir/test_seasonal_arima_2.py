import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import mylcgen as mlcg
#test seasonal arima models new 1/10/2018 
#why are these so poor forecaster. Not taking account of seasonal trends
#only first attempt. Try again tomorrow
#https://www.datasciencecentral.com/profiles/blogs/tutorial-forecasting-with-seasonal-arima


tlen = 3000.0
dt   = 1.0

#enter period of long time-scale variations
period = [365.0,180.0]
color= ['r','b']

#amplitude relative to standard deviation of random time series
amp_p  = [3.0,3.0]

#generate fake data with a large amplitude period



#first generate the random-ness
dat = mlcg.mylcgen(datfile='',p0=1.0,f0=0.1,a=-2,b=-2,tlo=0,thi=tlen,dt=dt,ploton=0,iseed=-1,meannorm = -1., sdnorm = -1.0)
nd  = np.shape(dat[:,0])[0]


std = np.std(dat[:,1])
#now modulate with a large (annual) like the bike hire data
variation = np.zeros(nd)
for ipn in range(len(period)):
 variation = variation + amp_p[ipn]*std*np.sin(2*np.pi/period[ipn] * dat[:,0])



#add on the periodic variability
dat[:,1] = dat[:,1] + variation


#cant have -ve values if taking log. Add on 2*,in value
dat[:,1] = dat[:,1] + np.abs(np.min(dat[:,1]))*2

#compute the fourier transform
ft = np.fft.fft(dat[:,1])
freq = np.fft.fftfreq(nd)
ps = np.abs(ft)**2




#plot the result
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(dat[:,0],dat[:,1],marker='o',ls=None)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Time Series')

yl = list(ax1.get_ylim())

ipn = 0
for pnow in period:
 tnow = pnow/4
 idp = 0
 while (tnow < dat[-1,0]):
  if (idp == 0):
   ax1.plot([tnow]*2,yl,ls='--',color=color[ipn],label='synthetic period '+np.str(pnow)+' days')
  else:
   ax1.plot([tnow]*2,yl,ls='--',color=color[ipn])
  tnow = tnow + pnow
  idp = idp + 1
 ipn = ipn + 1
plt.legend()
plt.savefig('test_data_diagnose.pdf')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(freq,ps,marker='o',ls=None)
ax1.set_xlabel('frequency (cycles/day)')
ax1.set_ylabel('P(f)')
ax1.set_xscale('log')
ax1.set_yscale('log')
yl = list(ax1.get_ylim())
for ipn in range(len(period)):
 ax1.plot([1./period[ipn]]*2,yl,ls='--',color=color[ipn],label='synthetic period '+np.str(period[ipn])+' days')
plt.legend()
plt.savefig('test_pspec_diagnose.pdf')



























#now perform the sarima forecasting
series           = pd.DataFrame(dat[:,1])
percent_training = 0.7
split_point      = np.int(round(len(series) * percent_training))
training, testing = series[0:split_point], series[split_point:]
 

#since the data is multiplicative apply log training (sarima can only do additive evolution?)
#training = np.log(training)


#since we have positive linear trend apply differencing of 1 period
training_diff = training.diff(periods=1).values[1:]



# we plot the residual log-differenced series
plt.plot(training_diff)
plt.title('Time series data')
plt.xlabel('years')
plt.ylabel('pasengers')





#import acf/pcf functions and run them with the differenced training data
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(training_diff,nlags=40)
lag_pacf = pacf(training_diff,nlags=40,method='ols')


#plot acf
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.stem(lag_acf)
plt.axhline(y=0,linestyle='-',color='black')
plt.axhline(y=-1.96/np.sqrt(len(training)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(training)),linestyle='--',color='gray')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.tight_layout()
plt.savefig('fig_acf_pacf.pdf')




#note the input is training not training_diff, if you input training_diff 
#then set d = 0

model = SARIMAX(training,order=(4,1,4),seasonal_order=(1,0,0,12),enforce_stationarity=False,
enforce_invertibility=False)
model_fit = model.fit(disp=False) #extract the fitted model






#now we can forecast
K = len(testing) #number of time steps we want to forecast (multi step forecasting)
forecast = model_fit.forecast(K) #apply the model to forecast out K time steps
forecast = np.exp(forecast) #transform back to the orriginal non-log scale

pred = model_fit.get_prediction(start = nd, end= nd+20)
ps = pred.summary_frame()
pslo = np.array(ps['mean_ci_lower'])
pshi = np.array(ps['mean_ci_upper'])



#plot forecast and display RMSE
plt.figure(figsize=(10,5))
plt.plot(forecast,'r')
plt.plot(series,'b')
plt.title('RMSE: %.2f'% np.sqrt(sum((forecast-testing)**2/len(testing))))
plt.xlabel('years')
plt.ylabel('Passengers')
plt.autoscale(enable=True,axis='x',tight=True)
plt.axvline(x=series.index[split_point],color='black') #line to divide training/test data
plt.savefig('forecast.pdf')





