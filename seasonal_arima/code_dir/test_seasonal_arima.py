#matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import mylcgen as mlc


thi = 100.0
dt = 1.0
tforecast = 20.0


dat = mlc.mylcgen(datfile='',p0=0.01,f0=0.1,a=-2,b=-2,tlo=0,thi=thi+tforecast,dt=dt,ploton=0,iseed=1,meannorm = -1., sdnorm = -1.0)
minsub = 2*np.abs(np.min(dat[:,1]))
dat[:,1] = dat[:,1] + minsub

idhi = np.where(dat[:,0] > thi)[0][0]
datfc = dat[idhi:,:]
dat = dat[:idhi,:]


ndat = np.shape(dat[:,0])[0]
t=dat[:,0]
xt=dat[:,1]


window = 12

startforecast = ndat
endforecast   = ndat+10

df=pd.DataFrame(xt, index=np.arange(ndat), columns=['riders'])
##load and prepare data
#df = pd.read_csv('portland-oregon-average-monthly-.csv', index_col=0)
#df.index.name=None
#df.reset_index(inplace=True)
#df.drop(df.index[114], inplace=True)
#
##convert time axis from datetime object to numerical value
#start = datetime.datetime.strptime("1973-01-01", "%Y-%m-%d")
#date_list = [start + relativedelta(months=x) for x in range(0,114)]
#df['index'] =date_list
#df.set_index(['index'], inplace=True)
#df.index.name=None
#
#
#df.columns= ['riders']
#df['riders'] = df.riders.apply(lambda x: int(x)*100)
#df.riders.plot(figsize=(12,8), title= 'Monthly Ridership', fontsize=14)
#plt.savefig('month_ridership.png', bbox_inches='tight')



#make a random walk light curve and test





#extract seasonal features from data to make it stationary
decomposition = seasonal_decompose(df.riders, freq=window)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)



#test for stationarity
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series.rolling(pd.Series(timeseries),window=window,center=False).mean()
    
    rolstd  = pd.Series.rolling(pd.Series(timeseries),window=window,center=False).std()
    #rolmean = pd.rolling_mean(timeseries, window=12)
    #rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    
    
    


test_stationarity(df.riders)


#test different models for stationarity
df.riders_log= df.riders.apply(lambda x: np.log(x))  
test_stationarity(df.riders_log)

df['first_difference'] = df.riders - df.riders.shift(1)  
test_stationarity(df.first_difference.dropna(inplace=False))

df['log_first_difference'] = df.riders_log - df.riders_log.shift(1)  
test_stationarity(df.log_first_difference.dropna(inplace=False))

df['seasonal_difference'] = df.riders - df.riders.shift(window)  
test_stationarity(df.seasonal_difference.dropna(inplace=False))

df['log_seasonal_difference'] = df.riders_log - df.riders_log.shift(window)  
test_stationarity(df.log_seasonal_difference.dropna(inplace=False))

df['seasonal_first_difference'] = df.first_difference - df.first_difference.shift(window)  
test_stationarity(df.seasonal_first_difference.dropna(inplace=False))

df['log_seasonal_first_difference'] = df.log_first_difference - df.log_first_difference.shift(window)  
test_stationarity(df.log_seasonal_first_difference.dropna(inplace=False))

#plot the auto correlation functions
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.seasonal_first_difference.iloc[window+1:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.seasonal_first_difference.iloc[window+1:], lags=40, ax=ax2)


#evaluate the models    
mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0), seasonal_order=(0,1,1,12))
results = mod.fit()
print results.summary()

mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
results = mod.fit()
print results.summary()





#perform forescast
df['forecast'] = results.predict(start = startforecast, end= endforecast, dynamic= True)  
ax1 = df[['riders', 'forecast']].plot(figsize=(12, 8)) 

#get condfidence limits
pred = results.get_prediction(start = startforecast, end= endforecast)
ps = pred.summary_frame()
pslo = np.array(ps['mean_ci_lower'])
pshi = np.array(ps['mean_ci_upper'])
pst  = np.arange(startforecast,endforecast+1)
psmean = np.array(ps['mean'])
ax1.plot(pst,psmean)
ax1.plot(datfc[:,0],datfc[:,1])

ax1.fill_between(pst,pslo,pshi,alpha=0.2)

plt.savefig('ts_df_predict.png', bbox_inches='tight')







#npredict =df.riders['1982'].shape[0]
#fig, ax = plt.subplots(figsize=(12,6))
#npre = 12
#ax.set(title='Ridership', xlabel='Date', ylabel='Riders')
#ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'riders'], 'o', label='Observed')
#ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'forecast'], 'g', label='Dynamic forecast')
#legend = ax.legend(loc='lower right')
#legend.get_frame().set_facecolor('w')
#plt.savefig('ts_predict_compare.png', bbox_inches='tight')
#
#start = datetime.datetime.strptime("1982-07-01", "%Y-%m-%d")
#date_list = [start + relativedelta(months=x) for x in range(0,12)]
#future = pd.DataFrame(index=date_list, columns= df.columns)
#df = pd.concat([df, future])
#
#df['forecast'] = results.predict(start = 114, end = 125, dynamic= True)  
#df[['riders', 'forecast']].ix[-24:].plot(figsize=(12, 8)) 
#plt.savefig('ts_predict_future.png', bbox_inches='tight')
    