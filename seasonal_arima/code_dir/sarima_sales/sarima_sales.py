#http://barnesanalytics.com/sarima-models-using-statsmodels-in-python

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


frac_fc = 0.95
frac_endfc = 1.2


df=pd.read_csv('salesdata.csv')




#df.index=pd.to_datetime(df['Date'])
df = df.convert_objects(convert_dates='coerce',convert_numeric=True)

#df['Sales'].plot()
#plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Sales'].diff().dropna(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Sales'].diff().dropna(), lags=40, ax=ax2)
#plt.show()

#This model is shown but not run because it will return an error.
#model=sm.tsa.statespace.SARIMAX(endog=df['Sales'],order=(0,1,0),seasonal_order=(0,1,0,12),trend='c',enforce_invertibility=False)
#results=model.fit()
#print(results.summary())

#To show you why it will return an error use this code:
print(df['Sales'].diff().diff(12))
#%%
np.random.seed(5967)
noise=[np.random.normal(scale=500)]

for i in range(len(df)-1):
    noise.append(np.random.normal(scale=500)+noise[i]*(-0.85))
df['Sales2']=df['Sales']+noise
df['Sales2'].plot()
#plt.show()



#%%
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Sales2'].diff().diff(12).dropna(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Sales2'].diff().diff(12).dropna(), lags=40, ax=ax2)
#plt.show()

nt = np.shape(df.values[:,0])[0]


startforecast = np.int(frac_fc*nt)#np.int(0.75*nt)
endforecast   = np.int(frac_endfc*nt)

model=sm.tsa.statespace.SARIMAX(endog=df['Sales2'][:startforecast],order=(1,1,0),seasonal_order=(0,1,0,12),trend='c',enforce_invertibility=False)
results=model.fit()
print(results.summary())
#%%
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(results.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(results.resid, lags=40, ax=ax2)
#plt.show()

df['noise']=noise
results.resid.loc['2008-02-01':].plot(label='Regression Residuals')
df['noise'].loc['2008-02-01':].plot(color='r',label='True Noise')
plt.legend(loc=2)
#plt.show()

#%%
model2=sm.tsa.statespace.SARIMAX(endog=df['Sales2'][:startforecast],order=(1,1,0),seasonal_order=(0,1,1,12),trend='c',enforce_invertibility=False)
results2=model2.fit()
print(results2.summary())

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(results2.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(results2.resid, lags=40, ax=ax2)
#plt.show()

df['noise']=[noise[i]+0.85*noise[i-1] if i>0 else 0 for i in range(len(noise))]
results2.resid.loc['2008-02-01':].plot(label='Regression Residuals')
df['noise'].loc['2008-02-01':].plot(color='r',label='True Noise')
plt.legend(loc=2)
#plt.show()
plt.clf()


#convert to time in days since 1st observation
idtrain_end = (pd.Timestamp(year=2017, month=9,day=1) - pd.Timestamp(year=2014, month=1, day=1)).days
print 'convert format'
ftot = df.convert_objects(convert_dates='coerce',convert_numeric=True)
print 'convert format - done'
t = ftot.values[:,0] - ftot.values[0,0]
time = np.array([tn.days for tn in t])
nt   = np.shape(time)[0]


#plot the forecast
fig = plt.figure()
ax1 = fig.add_subplot(111)



pred  = results.get_prediction(start = startforecast, end= endforecast )
ps    = pred.summary_frame()
pslo  = np.array(ps['mean_ci_lower'])
pshi  = np.array(ps['mean_ci_upper'])
npred = np.shape(pslo)[0]

dtime = time[1]-time[0]
tfstart = time[startforecast]
tfend   = tfstart+(endforecast-startforecast)*dtime

pst  = np.linspace(tfstart,tfend,npred)#time[startforecast-1:endforecast+1]
psmean = np.array(ps['mean'])
ax1.plot(pst,psmean,label='forecast')
ax1.fill_between(pst,pslo,pshi,label='forecast uncertainty',alpha = 0.3)
y = df[u'Sales2'].values
x = time
ax1.plot(x,y,label='actual')
plt.legend()
plt.savefig('fig_sarima_forecast.pdf')