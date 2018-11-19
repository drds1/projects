#python project to load in numbers spreadsheets for each 
#exp_xxxx.numbers file in /Users/david/projects/expenses_data
#seasonal arima model article https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/

import numpy as np
import matplotlib.pylab as plt
import glob
import pandas as pd
import datetime
import statsmodels.api as sm


frequency = 1./180 #fake 6 month signal
tlo = 0.0
thi = 365*5
tref = 0.0#thi/2
dt = 1.0

#generate synthetic time series for test
t = np.arange(tlo,thi,dt)
x = np.sin(2 * np.pi * frequency * t)

#add trend final -10.0 indicates amplitude at the end of the time sequence
grad = 1.0
y1   = 10.0
noiseamp = 0.5







y    = 0.0 + 0.0*(t-tref) + 1.e-5*(t-tref)**2 + 0.0*(t-tref)**3 
#grad*(x-t[-1]) + y1
nt = np.shape(t)[0]
noise = np.random.randn(nt)*noiseamp
signal = x + y + noise


#plot out th efake time series
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(t,signal)
plt.savefig('test_time_sequence.pdf')









##use data science to forecast future variability
#model=sm.tsa.statespace.SARIMAX(endog=dweek['sum'],order=orderin,seasonal_order=(0,1,0,12),trend='c',enforce_invertibility=False)
#
#results=model.fit()
#pred  = results.get_prediction(start = times[-1], end= times[-1]+(times[np.int(nsize*pc_forecast)]-times[0]) )
#
##set the confidence interval with the alpha_conf argument specified at the top
#ps    = pred.summary_frame(alpha=alpha_conf)
#pslo  = np.array(ps['mean_ci_lower'])
#pshi  = np.array(ps['mean_ci_upper'])
#npred = np.shape(pslo)[0]






























# #plot the time series 9can also use dweek.plot() for simple option but less customisation
# x = dweek.axes[0]
# y = dweek['sum']
# 
# if (combine == 0):
#  fig = plt.figure()
#  ax1 = fig.add_subplot(111)
# else:
#  ax1 = fig.add_subplot(np.int(np.ceil(nf/2.)),2,i+1)
#
# ax1.plot(x,y,label='data')
# 
# xmod = ps.axes[0]
# ymean = np.array(ps['mean'])
# ylo  = np.array(ps['mean_ci_lower'])
# yhi  = np.array(ps['mean_ci_upper'])
# ax1.plot(xmod,ymean,label='forecast')
# ax1.set_title(lab)
# ax1.fill_between(xmod,ylo,yhi,alpha = 0.3,label='uncertainty')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Expense GBP')
# plt.legend(fontsize='xx-small')
# 
# if (combine == 0):
#  plt.savefig('fig_'+lab+'.png')
#
#
#
#if (combine == 1):
# plt.tight_layout()
# plt.savefig('summary.png') 
# 