#load bike data

import numpy as np
import pandas as pd
import csv
import glob
import matplotlib.pylab as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.feature_extraction import FeatureHasher as fhe


#Note this file is only for ingesting the data and
#making the tie series and power spectrum plots
#I attempted to model the time series using a neural net
#but used the rw-mod model instead. Some of the text
#is therefore obsolete and commented out
#converting the dates into days-of-the-week is also
#unnecessary as the power spectrum analysis automatically 
#reveals trends operating on weekly timescales

start_end_id = [6184,6085]
recalc = 1
thin = 1#1/10th of input value to speed things up
daylims = [230,260]
reference_year = 2014
dowon = 0
time_step = 1.0

#Since the data 'stops' at the end of august, do not use times after this to train the neural net
#dont want to be unfair
final_day = (pd.Timestamp(year=2017, month=8, day=31) - pd.Timestamp(year=reference_year, month=1, day=1)).days



#small definintion to convert dates to number of days into year
def date_to_nth_day(date,refyear = -1):
    date = pd.to_datetime(date)
    if (refyear == -1):
     new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
     yearout = date.year
    else:
     new_year_day = pd.Timestamp(year=refyear, month=1, day=1)
     yearout = refyear
    
    return ((date - new_year_day).days + 1,yearout)
dnd = np.vectorize(date_to_nth_day)




#ingest all the data and save to files labels_info.csv and labels_station.csv for rw-mod modelling
#not all the onehot encoding is necessary (I tried a few things that didnt work)
if (recalc == 1):
 dsave = glob.glob('BixiMontrealRentals20*/')
 
 idc = 0
 
 for dnow in dsave:
  
  flist = glob.glob(dnow+'OD_*.csv')
  stat_file = glob.glob(dnow+'Station*.csv')[0]
  
  #load station data 
  stations = pd.read_csv(stat_file)
  if (idc == 0):
   stat_save = stations
  else:
   stat_save = pd.DataFrame.append(stat_save,stations)
 
  
  for fnow in flist:
   print 'reading bike data...',fnow
   features = pd.read_csv(fnow,sep=',',keep_default_na=False)
   if (thin > 0):
    idspec = features.index[(features[u'start_station_code'] == start_end_id[0]) & (features[u'end_station_code'] == start_end_id[1])]
    ndat = np.shape(features.values)[0]
    print 'selecting subsample 1/',thin,'   (original size',ndat,') for speed. Turn off for final results'
    idx = np.random.choice(np.arange(ndat), size=ndat/thin, replace=False, p=None) 
    idxtot = np.unique(np.concatenate([idx,np.array(idspec)]))
    features = features.iloc[idxtot]
    print 'done thinning'
   
   
   if (idc == 0):
    ftot = features
   else:
    ftot = pd.DataFrame.append(ftot,features)
   idc = idc + 1
 
 
 
 
 #remove duplicate entries from the station (stat_save) data frame
 stat_save = stat_save.drop_duplicates(subset=u'code')
 
 
 
 #convert the ftot data frame 'date' column from object to date_time object
 #... and all numerical quantities to floats
 print 'convert format'
 ftot = ftot.convert_objects(convert_dates='coerce',convert_numeric=True)
 print 'convert format - done'
 ndat,ndim = np.shape(ftot.values)
 
 #record day of the week
 print 'dow',ndat,ndim
 if (dowon == 1):
  dow = pd.to_datetime(ftot[u'start_date']).dt.weekday.values[:]#[np.int(ftot.values[i,0].weekday()) for i in range(ndat)]
 else:
  dow = np.zeros(ndat)
  
 print 'dow - done...assigning'
 ftot = ftot.assign(dow=dow)
 print 'dow - done assigning'
 #
 #record day into year
 print 'record day number',ndat,ndim
 a = ftot.values[:,0]
 a2 = dnd(a)
 day_count_year = a2[0]
 years = a2[1]
 years_included = np.unique(years)
 #day_count_year = [(ftot.values[i,0] - pd.Timestamp(year=ftot.values[i,0].year, month=1, day=1)).days for i in range(ndat)]
 ftot = ftot.assign(day_count_year=day_count_year)
 print 'record day number - done'
 
 #record day count relative to a reference year (e.g 2014 since start of observations)
 day_count_ref = dnd(a,refyear=reference_year)[0]
 ftot = ftot.assign(day_count_ref=day_count_ref)
 
 

 #save data frame to pickle to prevent having to recalculate all this on subsequent runs (saves time)
 print 'saving data'
 #ftot.to_pickle('ftot.pkl')
 #stat_save.to_pickle('stat_save.pkl')
 np.savetxt('years.txt',years)
 print 'done saving data'
 
else:
 years = np.array(np.loadtxt('years.txt'),dtype='int')
 ftot = pd.read_pickle('ftot.pkl')
 stat_save = pd.read_pickle('stat_save.pkl')
 years_included = np.unique(years)





#identify all trips from start to end station and save these as a separate csv file
fselect = ftot.loc[(ftot[u'start_station_code'] == start_end_id[0]) & (ftot[u'end_station_code'] == start_end_id[1])]
fselect.to_csv('selected_station.csv')





#visualise the data between the selected stations

#duration of trip
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = fselect.values[:,2] - fselect.values[:,0]
xp = [xn.seconds for xn in x]
sig = (100 - 68.27)/2
uncert = np.percentile(xp,[sig,50,100-sig])
ax1.hist(xp,histtype='step')
ylim = ax1.get_ylim()
ax1.plot([uncert[0]]*2,ylim,ls='--',color='b')
ax1.plot([uncert[1]]*2,ylim,ls='-',color='b')
ax1.plot([uncert[2]]*2,ylim,ls='--',color='b')
ax1.set_xlabel('Duration of hire')
plt.savefig('plot_duration.pdf')









#load a subset of data around the sep 1st to sep 8th window (1 months either side)
f_short_term = ftot.loc[(ftot[u'day_count_year'] > daylims[0]) & (ftot[u'day_count_year'] < daylims[1])]



#plot the time series average bikes rented across year split by latitude and members

print 'making histogram timeseries.pdf around required dates'
xhist = np.arange(daylims[0]-1.0,daylims[1]+2.0,time_step)

fig = plt.figure()
ax1 = fig.add_subplot(111)

print 'calculating histogram...'
y =np.array(f_short_term.values[:,-1],dtype='float')
yhist = ax1.hist(y,bins = xhist,histtype='step')
print 'done making histogram'
ax1.set_xlabel('Time of year')
ax1.set_ylabel('Total bikes hired')
plt.savefig('timeseries.pdf')


#make histogram of the day-of-week variability
##how the trend depends on time of year
xhist = np.arange(8)
y = np.array(ftot.values[:,-3],dtype='float')
yhist = np.histogram(y,bins=xhist)[0]
xhist = xhist[1:]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('days of week')
ax1.set_ylabel('Bikes hired')
ax1.plot(xhist,yhist)
plt.savefig('time_series_dow.pdf')

#make histogram of all data DO NOT combine years together and study long term
#year-to-year trends too
#identify 4th September for each of the years
title = ['time_series_ref2014_select_station.pdf','time_series_ref2014.pdf']
yn = [fselect,ftot]

nhist = len(yn)
tstampref = pd.Timestamp(year=reference_year, month=1, day=1)

polysave = []
for ih in range(nhist):
 xhist = np.arange(-1.0,367*4,time_step)
 xh = (xhist[1:] + xhist[:-1])/2
 xhist_dow = tstampref.weekday() + np.mod(xh,7)
 nxh = np.shape(xh)[0]
 day_ref = []
 idx_inc = []
 ymean_year = []
 
 
 for ynow in years_included:
  day_ref.append( (pd.Timestamp(year=ynow, month=9, day=4) - tstampref).days)
  idx_inc.append( np.where(years == ynow)[0])
 y = np.array(yn[ih].values[:,-1],dtype='float')
 
 
 
 
 yhist = np.histogram(y,bins=xhist)[0]
 
 if (ih == 0):
  labop = np.zeros((nxh,2))
  labop[:,0] = xh
  labop[:,1] = yhist
  np.savetxt('labels_station.csv',labop,delimiter=',')
 
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 ax1.set_xlabel('Days since January 1st 2014')
 ax1.set_ylabel('Bikes hired')
 ax1.plot(xh,yhist,zorder=1)
 yl = list(ax1.get_ylim())
 
 ic = 0
 xmean_year = []
 ymean_year = []
 ysd_year = []
 
 idlo = 0
 idhi = 0
 #idstart = np.where(yhist > 0)[0][0]
 #idend = idstart + np.where(yhist[idstart:]>0)[0][4]
 for ynow in day_ref:
  #calculate the mean daily bike hire for the year in question
  #idlo = idstart + ic*365#np.int(xhist[0])+ ic*365
  #idhi = idlo+365
  idlo = idhi + np.where(yhist[idhi:] > 0)[0][0]
  idhi = idlo + max(219,np.where(yhist[idlo:]==0)[0][4])
  print idlo,idhi,xh[idlo],xh[idhi]
  
  idinc = np.arange(idlo,idhi,1)
  
  xmean_year.append(np.nanmean(xh[idinc]))
  ymean_year.append(np.nanmean(yhist[idinc]))
  ysd_year.append(np.nanstd(yhist[idinc]))
  
 
  
  if (ic == 0):
   ax1.plot([ynow]*2,yl,color='r',ls='--',zorder=2,label='September 4th')
  else:
   ax1.plot([ynow]*2,yl,color='r',ls='--',zorder=2,label=None)

  ic = ic + 1

 ndeg =2 
 poly_co = np.polyfit(xmean_year, ymean_year, ndeg, rcond=None, full=False, w=None, cov=False)
 polysave.append(poly_co)
 y_poly_res = np.ones(nxh)*poly_co[-1] 
 print ''
 print 'xmean,ymean,ysd',title[ih]
 print xmean_year
 print ymean_year
 print ysd_year
 print ''
 #raw_input()
 for ip in range(ndeg):
  y_poly_res  = y_poly_res + poly_co[ip]*xh**(ndeg-ip)  
 plt.legend(loc=2)
 xl = list(ax1.get_xlim())
 ax1.errorbar(xmean_year,ymean_year,ysd_year,ls='',color='r',marker='o',zorder=3,label='Daily averages (over year)')
 ax1.plot(xh,y_poly_res,color='r',zorder=4,label='polynoimial fit')
 ax2 = ax1.twiny()
 ax2.set_xticks(day_ref)
 ax2.set_xticklabels(['04/09/'+np.str(cn) for cn in years_included])
 ax2.set_xlim(xl)
 ax1.legend()
 plt.savefig(title[ih])




#compute fraction of the bikes that are hired between the required two stations as a function of
fig = plt.figure()
ax1 = fig.add_subplot(111)


plt.savefig('frac_between_stations.pdf')




#save the polynomial co-efficients for use in the 
#scaling function S(t) = s(t)/g(t)
print 'saving polynomial co-efficients...'
polysave = np.array(polysave)
np.savetxt('polysave.txt',polysave)
print '... done'
print ''

#plot the power spectrum to visualise short-and-long-term variability
fig = plt.figure()
ax1 = fig.add_subplot(111)
ps = np.abs(np.fft.fft(yhist))**2
nyhist = yhist.size
freqs = np.fft.fftfreq(nyhist, time_step)
idx = np.argsort(freqs)
ax1.plot(freqs[idx],ps[idx])
ax1.set_xscale('log')
ax1.set_yscale('log')

ylim = list(ax1.get_ylim())
xlim = list(ax1.get_xlim())
#plot the expected 1 year periodic reference frequency
xrefs = [1./365,1./7]
xreflab =['1 year','1 week']
[ax1.plot([xn]*2,ylim,ls='--',color='r',label=None,zorder=0) for xn in xrefs]
ax2 = ax1.twiny()
ax2.set_xscale('log')
ax2.set_xlim(xlim)
ax2.set_xticks(xrefs)
ax2.set_xticklabels(xreflab)

ax1.set_xlabel('frequency (cycles/day)')
ax1.set_ylabel('P(f) Power spectrum')
plt.savefig('power_spectrum.pdf')








#save the data for modellng in rw-mod.py

#convert dow to binary format using one-hot encoding 

fnew = pd.get_dummies(xhist_dow)
col_lab = fnew.columns
f = np.array(fnew)
ny,nx = np.shape(f)
dat = np.zeros((ny,nx+1))
dat[:,0] = xh
dat[:,1:] = f


#we want to include the day number (from the reference) capture the year periodicity
#and the day of the week to capture the weekly periodicity
labels = yhist


#save the labels info to test in recurrent neural network
labop = np.zeros((ny,2))
labop[:,1] = labels
labop[:,0] = xh
np.savetxt('labels_info.csv',labop,delimiter=',')

#save the labels for the selected data


##from sklearn.model_selection import train_test_split
##train_f, test_features, train_l, test_labels = train_test_split(dat, labels, test_size = 0.25)
#idtrain = np.where(xh < final_day)[0]
#idtest = np.where(xh > final_day)[0]
#train_f = dat[idtrain,:]
#train_l = labels[idtrain]
#test_f  = dat[idtest,:]
#test_l  = labels[idtest]
#
#from sklearn.neural_network import MLPRegressor
##from sklearn.ensemble import RandomForestRegressor
##rf = RandomForestRegressor(n_estimators= 1000, random_state=42)
##rf.fit(train_f, train_l)
##predictions_rf = rf.predict(test_f)
#
#
#
#reg = MLPRegressor(
#    hidden_layer_sizes=(200,),  activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
#    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#op = reg.fit(train_f, train_l)    
#
#op_test = reg.predict(test_f)
#
#op_test_all = reg.predict(train_f)
#
#
#
##make plot of trained neural network
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.set_xlabel('Days since January 1st 2014')
#ax1.set_ylabel('Bikes hired')
#ax1.plot(train_f[:,0],train_l,label='training data')
#ax1.plot(train_f[:,0],op_test_all,label='nn training data')
#
#ax1.plot(test_f[:,0],test_l,label='test data')
#ax1.plot(test_f[:,0],op_test,label='Neural net predictions')
##ax1.plot(test_f[:,0],predictions_rf,label='rf predictions')
#ax1.errorbar(xmean_year,ymean_year,ysd_year,ls='',color='r',marker='o',zorder=3,label='Daily averages (over year)')
#ax1.plot(xh,y_poly_res,color='r',zorder=4,label='polynoimial fit')
#ax2 = ax1.twiny()
#ax2.set_xticks(day_ref)
#ax2.set_xticklabels(['04/09/'+np.str(cn) for cn in years_included])
#ax2.set_xlim(xl)
#ax1.legend()
#
#plt.savefig('hist_posttrain.pdf')
#
#
##now have dat and labels saved have everything needed for neural net
#
##if machine learning mlp doesnt work just fit time series model with a power law power
##spectrum that mathces the computed power spectrum with extra compoents at 1 week and 1 year
##e.g f(t) = sk sin(wt) + ck cos(wt) + S_w sin(w_week t) + C_w cos(w_week t) + S_y sin(w_year t) + C_y cos(w_year t)
#
#
#
##test neural net regressor here
#x = np.arange(1,100,0.01)
#y = 2.*x**2 + 0.4*x +31.3
#
#xtest = np.array([13.6,29.4,67.4,89.0,104.6,130.5])
#ytest = 2.*xtest**2 + 0.4*xtest +31.3
#
#
##from sklearn.ensemble import RandomForestRegressor
##rf = RandomForestRegressor(n_estimators= 1000, random_state=42)
## Train the model on training data
##rf.fit(x.reshape(-1,1), y)
#
#
#
##!!!!!!!!!! MAKE PREDICTIONS ON TEST DATA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##!!!!!!!!!! DETERMINE PERFORMANCE METRICS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##predictions_rf = rf.predict(xtest.reshape(-1,1))
#
#
#regtest = MLPRegressor(
#    hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
#    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
##regtest = MLPRegressor(solver='sgd',activation='tanh', max_iter = 1000,alpha=1e-5,hidden_layer_sizes=(5,2), random_state=None)
#regtest.fit(x.reshape(-1,1), y)    
#optest = regtest.predict(xtest.reshape(-1,1))
#optest_all = regtest.predict(x.reshape(-1,1))
#
#
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(x,y,label='train',zorder=5)
#ax1.plot(xtest,ytest,label='test',marker='o',zorder=3)
#ax1.plot(xtest,optest,label='nn results test',marker='o',zorder=4)
##ax1.plot(xtest,predictions_rf,label='rf results test',marker='o',zorder=4)
#ax1.plot(x,optest_all,color='k',label='nn results all',zorder=6)
#plt.legend()
#plt.savefig('fig_nn_experiment.pdf')
#
#
#
#
#
#
#