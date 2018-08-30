import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from mylcgen import *
import myresample as mrs
from myrandom import *
import myfitrw_092018 as mfrw
import pandas as pd

#labels_info.csv,labels_station.csv
tit_input = 'labels_info.csv'
niterations = 1
fake = 0
look_back = 1

# fix random seed for reproducibility
numpy.random.seed(7)

idtrain_end = (pd.Timestamp(year=2017, month=9,day=1) - pd.Timestamp(year=2014, month=1, day=1)).days


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)






# if testing with fake data do that here
if (fake == 1):
 datpre      = mylcgen(tlo=0,thi=100,dt=0.1,iseed=132423)
 npre     = np.shape(datpre[:,0])[0]
 datmean  = np.std(datpre[:,1])
 snow =  np.ones(npre)/10*datmean 
 dat = datpre#mrs.myresample(dir='',fname=[''],dtave=1.0,sampmin=0.8,sampcode=3,datin=np.array((datpre[:,0],datpre[:,1],snow)).T)
 ndat = np.shape(dat[:,0])[0]
 sig = snow
 for i in range(ndat):
  dat[i,1] = normdis(1,dat[i,1],sig[i])[0]
 dataset = dat[:,1].reshape(-1,1)
 idmax = int(0.8*ndat)
 a = mfrw.fitrw([dat[:idmax,0]],[dat[:idmax,1]],[sig[:idmax]],floin=1./200,fhiin=2.0,ploton=1,dtresin=-1,nits = 1,tplotlims=[-10.0,120.0,0.1])
# load the dataset
else:
 dataframe = read_csv(tit_input, usecols=[1], engine='python', skipfooter=3)
 dataset = dataframe.values
 dataset = dataset.astype('float32')



# normalize the dataset
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = idtrain_end#int(len(dataset) * 0.09)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='logcosh', optimizer='Adagrad')#logcosh Adagrad,logcosh rmsprop
model.fit(trainX, trainY, epochs=niterations, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


ydat = scaler.inverse_transform(dataset)[:,0]
ypred = trainPredictPlot[:,0]
yfore = testPredictPlot[:,0]
#perform iterated optimal scaling to try to reconcile scaling problem

C = 0
A = 1
n_ios = 30

ydmean = np.nanmean(ydat)
ydrms = np.nanstd(ydat)
yd = ydat#(ydat - ydmean)/ydrms

yprms = np.nanstd(ypred)
ypmean = np.nanmean(ypred)
yp = ypred#(ypred - ypmean)/yprms


test = (yd - A*yp)
idx = np.where(test == test)[0]
nid = np.shape(idx)[0]
for i in range(n_ios):
 A = np.nansum((yd - C)*yp)/np.nansum(yp**2)
 C = np.nansum(yd - A*yp)/nid
 #C = -np.nanmin(yp)
 #print np.nansum((ydat - C)*ypred),np.nansum(ypred**2),np.nansum((ydat - C)*ypred)/np.nansum(ypred**2)
 #print np.nansum(ydat - A*ypred),np.nansum(ydat),np.nansum(A*ypred)
 print 'ios iteration ',i,'  A=',A,' C=',C,' rmserror',np.nansum(((A*yp+C)-yd)**2),' max model',A*np.nanmax(yp) + C 
 #raw_input()

#A=1.1


#add legend that is september 4th for each year
years = [2014,2015,2016,2017]
test_day = [(pd.Timestamp(year=ynow, month=9, day=4) - pd.Timestamp(year=2014, month=1, day=1)).days for ynow in years]


xlim = [[0,1500],[test_day[-1] - 20,test_day[-1]+27]]

nplot = len(xlim)
# plot baseline and predictions

nyp = np.shape(yp)[0]
xdat = np.arange(nyp)
xpre = xdat - 1.0

for iplot in range(nplot):
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 
 ax1.plot(xdat,yd,label='Actual')

 ax1.plot(xpre,A*testPredictPlot+C,label='Predicted forecast')
  
 if (iplot == 0):
  ax1.plot(xpre,A*yp+C,label='Predicted')
 
 ax1.set_xlabel('Days since January 1st 2014')
 ax1.set_ylabel('Bikes hired')
 
 
 #xlim = list(ax1.get_xlim())
 ylim = list(ax1.get_ylim())
 ax1.set_xlim(xlim[iplot])
 
 if (iplot == 1):
  ax2 = ax1.twiny()
  ax2.set_xlim(xlim[iplot])
  ax2.set_xticks(test_day)
  ax2.set_xticklabels(['04/09/'+np.str(ynow) for ynow in years])
 
 if (iplot == 0):
  for id in range(len(years)):
   td = test_day[id]
   if (id == 0):
    ax1.plot([td]*2,ylim,color='r',ls='--',label='September 4th')
   else:
    ax1.plot([td]*2,ylim,color='r',ls='--',label=None)
  
 elif (iplot == 1):
  xp = [(pd.Timestamp(year=ynow, month=9, day=4) - pd.Timestamp(year=2014, month=1, day=1)).days,
  (pd.Timestamp(year=ynow, month=9, day=12) - pd.Timestamp(year=2014, month=1, day=1)).days]
  ax1.plot([xp[0]]*2,ylim,color='r',ls='--',label='September 4th to September 11th')
  ax1.plot([xp[1]]*2,ylim,color='r',ls='--',label=None)
 ax1.legend()
 plt.savefig('fig_train_test_rnn_'+np.str(iplot)+'.pdf')
 
 




#if (fake == 1):
# model.save('timeseries_keras_model.h5')
# #load the model to make new prediction (just to check the code isnt faking it!)
# modload = load_model('timeseries_keras_model.h5')
 







from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 

idtrain_end = (pd.Timestamp(year=2017, month=9,day=1) - pd.Timestamp(year=2014, month=1, day=1)).days
idaug4 = (pd.Timestamp(year=2017, month=9,day=4) - pd.Timestamp(year=2014, month=1, day=1)).days
test = [x for x in testX[:100,0]]
history = [x for x in trainX[:,0]]
#undo the recurrent neural net transformation
history = scaler.inverse_transform(history)
test = scaler.inverse_transform(test)


ntest = len(test)
nhis = len(history)
xp = np.arange(ntest+nhis)
predictions = list()

print 'PROBLEM YOU R ARMA MODEL IS UPDATING ITSELF WITH THE PREVIOUS DAY EVERY LOOP!'
print 'cant append to the history in this loop not allowed!'
#raw_input()
model = ARIMA(history, order=(5,1,0))
model_fit = model.fit(disp=0)
sig = []

a = model_fit.forecast(steps=ntest)
predictions = a[0]
sig_p = a[1]
#for i in range(ntest):
# 
# predictions.append(model_fit.forecast(steps=i+1)[0][0] )#model_fit.predict(start=nhis,end=nhis+ntest-1)
#history_new = list(history)
history = np.append(history[:,0],test[:,0])
#predictions = 

#for t in range(len(test)):
#	model = ARIMA(history, order=(5,1,0))
#	model_fit = model.fit(disp=0)
#	output = model_fit.forecast()
#	yhat = output[0]
#	predictions.append(yhat)
#	obs = test[t]
#	history.append(obs)
#	print('predicted=%f, expected=%f' % (yhat, obs))
#	print 'nhis len history',nhis,len(history)
#error = mean_squared_error(test, predictions)
#print('Test MSE: %.3f' % error)
# plot

xlim = [[xdat[0],xdat[-1]],[xdat[idtrain_end]-10.,xdat[idaug4]+20]]
nplot = len(xlim)

#evaluate the scaling function using the polynomial fits performed in 
#main_code.py



#if we are plotting the specific station analysis need to load the global fits from previous simulation
#if (tit_input == 'labels_station.csv'):
# p = np.loadtxt('polysave.txt')
# sp = p[0,-1::-1]
# gp = p[1,-1::-1]
# 
# gt = gp[0] + xp*gp[1] + gp[2]*xp**2 
# st = sp[0] + xp*sp[1] + sp[2]*xp**2 
#
#
# St = st/gt
# predictions = np.loadtxt('predictions.txt')*St[nhis:]
# #xdat = np.loadtxt('xdat.txt')
#else:
# pass
# 
 
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






#use random walk fittingtplotlims[0],tplotlims[1],tplotlims[2]
nxp = np.shape(xp)[0]
parmout,covout,freq,tplot,xplot,xplotsd = mfrw.fitrw([xp[:nhis]],[history[:nhis]],
[np.ones(nhis)],extra_f=[1./365.,1.0/7],nits=100,tplotlims=[0.0,1450,1.0])



po =parmout[0]
dtplot = np.mean(tplot[1:] - tplot[:-1])
tplot_extra = 1.*tplot#np.append(tplot,np.arange(tplot[-1]+dtplot,tplot[-1]+dtplot + 100.0, dtplot))
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
 plt.savefig('drw_custom_total_ts_'+np.str(iplot)+'.pdf')


np.savetxt('history.txt',history)
np.savetxt('test.txt',test)
np.savetxt('predictions_drw_station.txt',np.array((tplot[idend_t:],xplot[idend_t:],xplotsd[idend_t:])))


