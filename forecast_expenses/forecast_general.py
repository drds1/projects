#python project to load in numbers spreadsheets for each 
#exp_xxxx.numbers file in /Users/david/projects/expenses_data
#seasonal arima model article https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/

import numpy as np
import matplotlib.pylab as plt
import glob
import pandas as pd
import datetime
import statsmodels.api as sm

combine = 0
frequency = 1./30. #fake 6 month signal
f2 = 1./60
tlo = 0.0
thi = 365
tref = 0.0#thi/2
dt = 1.0
lab = ''
amp1 = 5.0
amp2 = 10.0
amps = [amp1,amp2]
freqs = [frequency,f2]
poly = [0,0,5.e-5,0.0]
tfclo = thi
tfchi = thi + 365
#generate synthetic time series for test
t = np.arange(tlo,thi,dt)
x = 5*np.sin(2 * np.pi * frequency * t) + 10*np.sin(2 * np.pi * f2/2 * t)
#add trend final -10.0 indicates amplitude at the end of the time sequence
grad = 1.0
y1   = 10.0
noiseamp = 0.5

pc_forecast = 0.4 #forecast ahead 40% the original length of the time series

#specify the confidence interval with the alpha argument 
#e.g alpha = 0.05 is the 95pc confidence interval (2 sigma)
#alpha = 0.32 is the 1 sigma confidence interval (68pc)
alpha_conf = 0.32#0.05



def trend(t,poly,tref,freqs,amps):
 nt = np.shape(t)[0]
 npoly = len(poly)
 namps = len(amps)
 xtot = []
 for i in range(nt):
  trend = np.sum([poly[ip]*(t[i]-tref)**ip for ip in range(npoly)] )
  seasonal = np.sum([amps[ip]*np.sin(2*np.pi*freqs[ip]*t[i]) for ip in range(namps)] )
  xtot.append(trend + seasonal)

 return(np.array(xtot))
 
 

signal = trend(t,poly,tref,freqs,amps) 
#grad*(x-t[-1]) + y1
nt = np.shape(t)[0]
noise = np.random.randn(nt)*noiseamp
signal = signal + noise



tfull = np.arange(t[0],tfchi,dt)
xfull = trend(tfull,poly,tref,freqs,amps)#5*np.sin(2 * np.pi * frequency * tfull) + 10*np.sin(2 * np.pi * f2/2 * tfull)

#plot out th efake time series
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(t,signal)
plt.savefig('test_time_sequence.pdf')






nsteps = np.int(nt*pc_forecast)



#import pandas as pd
#df=pd.read_csv('/Users/david/projects/github_datascience/projects/seasonal_arima/code_dir/sarima_sales/salesdata.csv')
#
#signal = np.array(df['Sales'])
#nt = signal.size
#nsteps = np.int(nt*pc_forecast)
#
#t = np.arange(nt)
#dt = t[2] - t[1]
#tfclo = t[-1]
#tfchi = tlo + nsteps*dt
#print 'tlo,thi',tlo,thi





nfclo = np.int(tfclo/dt)
nfchi = np.int(tfchi/dt)

#use statsmodels api to sarima model to forecast future variability
orderin = (1,1,0)
model   = sm.tsa.statespace.SARIMAX(endog=signal,order=orderin,seasonal_order=(0,1,0,300),trend='c',enforce_invertibility=False)
results = model.fit()
pred    = results.get_prediction(start = nfclo, end= nfchi )
ps      = pred.summary_frame(alpha=alpha_conf)
pslo    = np.array(ps['mean_ci_lower'])
pshi    = np.array(ps['mean_ci_upper'])
npred   = np.shape(pslo)[0]












#plot the time series 9can also use dweek.plot() for simple option but less customisation
x = t
y = signal

if (combine == 0):
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
else:
 ax1 = fig.add_subplot(np.int(np.ceil(nf/2.)),2,i+1)

ax1.plot(x,y,label='data')

np.savetxt('test_timeseries.dat',y)

xmod = np.arange(tfclo,tfchi,dt)#ps.axes[0]
ymean = np.array(ps['mean'])

nym = np.shape(ymean)[0]
xmod = np.linspace(tfclo,tfchi,nym)

ylo  = np.array(ps['mean_ci_lower'])
yhi  = np.array(ps['mean_ci_upper'])
ax1.plot(xmod,ymean,label='forecast')
ax1.set_title(lab)
ax1.fill_between(xmod,ylo,yhi,alpha = 0.3,label='uncertainty')
ax1.plot(tfull,xfull,alpha=0.4,ls='--')
ax1.set_xlabel('Date')
ax1.set_ylabel('Expense GBP')
plt.legend(fontsize='xx-small')

if (combine == 0):
 plt.savefig('fig_'+lab+'.png')

#







def sarima(lc1,orderin=(1,1,0),seasonal_order = (0,1,0,120),trend='c',enforce_invertibility=False,
alpha_confidence=0.32,time_forecast=10.0,diagnostic_plot=''):

 time   = lc1[:,0]
 signal = lc1[:,1]
 dt = np.mean(time[1:]-time[:-1])
 nforecast = np.int(time_forecast/dt)
 ntime = np.shape(time)[0]
 tfclo = time[ntime-1]
 tfchi = time_forecast_lo + nforecast*dt
 idx_forecast_lo = ntime
 idx_forecast_hi = ntime + nforecast
 
 #fit the sarima model
 model   = sm.tsa.statespace.SARIMAX(endog=signal,
 order=order,
 seasonal_order=seasonal_order,
 trend='c',
 enforce_invertibility=False)
 results = model.fit()
 pred    = results.get_prediction(start = idx_forecast_lo, end= idx_forecast_hi )
 ps      = pred.summary_frame(alpha=alpha_confindence)
 pslo    = np.array(ps['mean_ci_lower'])
 pshi    = np.array(ps['mean_ci_upper'])
 npred   = np.shape(pslo)[0]
 


 #output the forecast time series
 ymean = np.array(ps['mean'])
 ylo  = np.array(ps['mean_ci_lower'])
 yhi  = np.array(ps['mean_ci_upper'])
 nym = np.shape(ymean)[0]
 xmod = np.linspace(tfclo,tfchi,nym)

  

 #plot the time series 9can also use dweek.plot() for simple option but less customisation
 if (diagnostic_plot != ''):
  x = time
  y = signal
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.plot(x,y,label='data')
  ax1.plot(xmod,ymean,label='forecast')
  ax1.set_title(lab)
  ax1.fill_between(xmod,ylo,yhi,alpha = 0.3,label='uncertainty')
  ax1.set_xlabel('Time')
  ax1.set_ylabel('light curve')
  
  plt.legend(fontsize='xx-small')
  if (disgnostic_plot != 'show'):
   plt.savefig(diagnostic_plot)
   plt.clf()
  else:
   plt.show()
   
   
 return(xmod,ymean,ylo,yhi)

