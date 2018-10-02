import numpy as np
import myfitrw_092018 as mfr
import mylcgen as mlcg
import matplotlib.pylab as plt

import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

tlen = 3000.0
dt   = 1.0

#enter period of long time-scale variations
period = [365.0,180.0]
color= ['r','b']

#amplitude relative to standard deviation of random time series
amp_p  = [0.1,0.1]

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








#mfr.fitrw([dat[:,0]],[dat[:,1]],[np.ones(nd)],floin=-1,fhiin=-1,plot_tit='fig_myrwfit',dtresin=-1,nits = 1,tplotlims=[],extra_f=[],
#p0=-1,bpl = [0.5,2,2],messages=0)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(dat[:,1], freq=10)  
trend         = decomposition.trend
seasonal      = decomposition.seasonal
res           = decomposition.resid
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)
plt.savefig('arima.pdf')




startforecast = 2600
endforecast   = 3000

mod = sm.tsa.statespace.SARIMAX(dat[:startforecast,1], trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
results = mod.fit()

plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
#get condfidence limits
pred = results.get_prediction(start = startforecast, end= endforecast)
ps = pred.summary_frame()
pslo = np.array(ps['mean_ci_lower'])
pshi = np.array(ps['mean_ci_upper'])
pst  = np.arange(startforecast,endforecast+1)
psmean = np.array(ps['mean'])
ax1.plot(pst,psmean)
ax1.plot(dat[:startforecast,0],dat[:startforecast,1])

ax1.fill_between(pst,pslo,pshi,alpha=0.2)

ax1.plot(dat[startforecast:,0],dat[startforecast:,1],ls='--',color='k')

plt.savefig('forecast.pdf')




parmout,covout,freq,tplot,xplot,xplotsd,p0_out,w0,dw,sig2_prior=mfr.fitrw([dat[:startforecast,0]],[dat[:startforecast,1]],[0.95],
floin=1./6000,fhiin=1.0,plot_tit='fig_myrwfit',dtresin=dat[:,0],nits = 100,tplotlims=[],extra_f=[1./365,1./180.],p0=-1,bpl = [0.5,2,2],
messages=0,prior = 1.0)
#mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
#results = mod.fit()
#print results.summary()

#this is a good tutorial follow for timeseries forecasting
#https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b



#compare the seasonal arima and 



