import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import myfitrw_092018 as mfrw
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 

#labels_info.csv,labels_station.csv

title_load = ['labels_info.csv','labels_station.csv']


for iall in range(len(title_load)):
 tit_input = title_load[iall]
 niterations = 1
 fake = 0
 look_back = 1
 
 # fix random seed for reproducibility
 numpy.random.seed(7)
 
 #define end of observations
 idtrain_end = (pd.Timestamp(year=2017, month=9,day=1) - pd.Timestamp(year=2014, month=1, day=1)).days
 
 
 # convert an array of values into a dataset matrix
 def create_dataset(dataset, look_back=1):
 	dataX, dataY = [], []
 	for i in range(len(dataset)-look_back-1):
 		a = dataset[i:(i+look_back), 0]
 		dataX.append(a)
 		dataY.append(dataset[i + look_back, 0])
 	return numpy.array(dataX), numpy.array(dataY)
 
 
 
 
 
 
 #load the data
 dataframe = read_csv(tit_input, usecols=[1], engine='python', skipfooter=3)
 dataset = dataframe.values
 dataset = dataset.astype('float32')
 
 xdat = np.arange(len(dataset))
 train_size = idtrain_end#int(len(dataset) * 0.09)
 test_size = len(dataset) - train_size
 trainX, testX = dataset[0:train_size,:][:,0], dataset[train_size:len(dataset),:][:,0]
 
 
 
 idtrain_end = (pd.Timestamp(year=2017, month=9,day=1) - pd.Timestamp(year=2014, month=1, day=1)).days
 idaug4 = (pd.Timestamp(year=2017, month=9,day=4) - pd.Timestamp(year=2014, month=1, day=1)).days
 test = [x for x in testX[:100]]
 history = [x for x in trainX[:]]
 
 
 
 
 #fit arma model and make plots
 ntest = len(test)
 nhis = len(history)
 xp = np.arange(ntest+nhis)
 predictions = list()
 
 model = ARIMA(history, order=(5,1,0))
 model_fit = model.fit(disp=0)
 sig = []
 
 a = model_fit.forecast(steps=ntest)
 predictions = a[0]
 sig_p = a[1]
 
 history = np.append(history[:],test[:])
 xlim = [[xdat[0],xdat[-1]],[xdat[idtrain_end]-10.,xdat[idaug4]+20]]
 nplot = len(xlim)
 
 
 for iplot in range(nplot):
  plt.clf()
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.plot(xp[:],history,label='history')
  ax1.plot(xp[nhis:],test,label='actual')
  ax1.plot(xp[nhis:]-1.0,predictions, color='red',label='predicted')
  ax1.fill_between(xp[nhis:]-1.0,predictions-sig_p,predictions+sig_p,alpha=0.4, color='red',label='predicted')
  ylim = list(ax1.get_ylim())
  ax1.plot([xdat[idaug4]]*2,ylim,label='4th - 11th September',ls='--',color='r')
  ax1.plot([xdat[idaug4]+7]*2,ylim,label=None,ls='--',color='r')
  ax1.set_xlim(xlim[iplot])
  plt.legend()
  ax1.set_xlabel('Days since January 1st 2014')
  ax1.set_ylabel('Bikes hired')
  plt.savefig('arma_total_ts_'+np.str(iplot)+'.pdf')
 
 np.savetxt('history.txt',history)
 np.savetxt('test.txt',test)
 np.savetxt('predictions.txt',predictions)
 #np.savetxt('predictions_arma_station.txt',np.array((xp[nhis:]-1.0,predictions,sig_p)))
 
 
 xop = xp[nhis:]-1.0
 yop = predictions
 sigop = sig_p
 ntplot = np.shape(xop)[0]
 a = [pd.Timestamp(year=2014, month=1, day=1) + pd.Timedelta(np.str(xop[i])+' days') for i in range(ntplot)]
 astr = [anow.strftime('%Y-%m-%d') for anow in a]
 nop = len(astr)
 f = open(np.str(iall)+'_predictions_arma.txt','w')
 for i in range(nop):
  f.write(astr[i]+'	'+np.str(xop[i])+'	'+np.str(sigop[i])+'\n')
 f.close()
  
 
 
 
 
 #use random walk fitting and make plots need myfitrw_092018.py
 nxp = np.shape(xp)[0]
 parmout,covout,freq,tplot,xplot,xplotsd = mfrw.fitrw([xp[:nhis]],[history[:nhis]],
 [np.ones(nhis)],extra_f=[1./365.,1.0/7],nits=100,tplotlims=[0.0,1450,1.0])
 
 
 
 po =parmout[0]
 dtplot = np.mean(tplot[1:] - tplot[:-1])
 tplot_extra = 1.*tplot
 ntp = np.shape(tplot_extra)[0]
 
 xplot_extra = np.zeros(ntp) + po[0]
 for it in range(ntp):
  tp = tplot_extra[it]
  xplot_extra[it] = po[0]+np.sum( po[1::2]*np.cos(2*np.pi*freq*tp) + po[2::2]*np.sin(2*np.pi*freq*tp) )
 
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 ax1.plot(xp,history)
 ax1.plot(tplot_extra,xplot_extra)
 plt.savefig('final_plot.png')
 
 idend_t = np.where(tplot_extra > idtrain_end)[0][0]
 ida4_t = np.where(tplot_extra > idaug4)[0][0]
 for iplot in range(nplot):
  plt.clf()
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.plot(xp[:],history,label='history')
  ax1.plot(xp[nhis:],test,label='actual')
  ax1.plot(tplot_extra,xplot_extra, color='red',ls=':',label='predicted')
  ax1.fill_between(tplot[idend_t:],xplot[idend_t:]-xplotsd[idend_t:],xplot[idend_t:]+xplotsd[idend_t:],alpha=0.4, color='red',label='predicted')
  ylim = list(ax1.get_ylim())
  ax1.plot([xdat[idaug4]]*2,ylim,label='4th - 11th September',ls='--',color='r')
  ax1.plot([xdat[idaug4]+7]*2,ylim,label=None,ls='--',color='r')
  ax1.set_xlim(xlim[iplot])
  plt.legend()
  ax1.set_xlabel('Days since January 1st 2014')
  ax1.set_ylabel('Bikes hired')
  plt.savefig(np.str(iall)+'_drw_custom_total_ts_'+np.str(iplot)+'.pdf')
 
 
 np.savetxt('history.txt',history)
 np.savetxt('test.txt',test)
 
 
 ntplot = np.shape(tplot)[0]
 a = [pd.Timestamp(year=2014, month=1, day=1) + pd.Timedelta(np.str(tplot[idend_t+i])+' days') for i in range(ntplot-idend_t)]
 astr = [anow.strftime('%Y-%m-%d') for anow in a]
 nop = len(astr)
 f = open(np.str(iall)+'_predictions_date.txt','w')
 for i in range(nop):
  f.write(astr[i]+'	'+np.str(xplot[idend_t+i])+'	'+np.str(xplotsd[idend_t+i])+'\n')
 f.close()
 
 #np.savetxt('predictions_drw_station_date.txt',np.array((a,xplot[idend_t:],xplotsd[idend_t:])))
 

