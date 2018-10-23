#python project to load in numbers spreadsheets for each 
#exp_xxxx.numbers file in /Users/david/projects/expenses_data

import numpy as np
import matplotlib.pylab as plt
import glob
import pandas as pd
import datetime
import statsmodels.api as sm
dir_input = '/Users/david/projects/expenses_data'
#load each file in the spreadsheet directory
files = glob.glob(dir_input + '/exp_*.csv')*4


#percentage time to forecast (e.g a 0.1 of a 100 day time series would forecast to 110 days)
pc_forecast = 0.4

#specify the confidence interval with the alpha argument 
#e.g alpha = 0.05 is the 95pc confidence interval (2 sigma)
#alpha = 0.32 is the 1 sigma confidence interval (68pc)
alpha_conf = 0.05


#combine into one figure
combine = 1




nf = len(files)

if (combine == 1):
 fig = plt.figure()

for i in range(nf):
 fnow = files[i]
 dnow = pd.read_csv(fnow)
 
 
 #plot title
 idlo = fnow.find('exp_')+4
 idhi = fnow.find('.csv')
 lab  = fnow[idlo:idhi]
 
 #isolate the dates (use dayfirst=True to assume UK date format),
 #example https://stackoverflow.com/questions/26763344/convert-pandas-column-to-datetime
 #2nd line sorts into ascending time order
 
 dnow['Date'] = pd.to_datetime(dnow['Date'],dayfirst=True)
 dnow = dnow.sort_values(by=['Date'])
 dates = dnow.values[:,0]
 
 #find the start date to act as reference
 date_min = min(dates)
 
 #subtract the min date and convert to days
 dr      = [datenow - date_min for datenow in dates]
 dr_days = np.array([drnow.days for drnow in dr])
 
 #sort into ascending order if not already sorted
 idsort = np.argsort(dr_days)
 
 #group the data weekly and compute the total expenditure at the end of each Sunday
 # example here https://stackoverflow.com/questions/45281297/group-by-week-in-pandas
 # another example of grouping by week
 # https://stackoverflow.com/questions/45281297/group-by-week-in-pandas
 dweek = dnow.groupby(pd.Grouper(freq='W-Sun', key='Date'))['Amount'].agg(['sum'])
 tots  = dweek.values[:,0]
 times = dweek.axes[0]
 nsize = dweek.size
 nforecast = 10
 
 
 #use aic to test many different models
 import statsm_aic_test as st
 #op=st.aic_test(dweek['sum'],order=[3,3,3])
 orderin = (1,1,0)#np.array(op[0,1:],dtype='int')
 #use data science to forecast future variability
 model=sm.tsa.statespace.SARIMAX(endog=dweek['sum'],order=orderin,seasonal_order=(0,1,0,12),trend='c',enforce_invertibility=False)
 
 results=model.fit()
 results.conf_int(alpha=0.05)
 pred  = results.get_prediction(start = times[-1], end= times[-1]+(times[np.int(nsize*pc_forecast)]-times[0]) )
 
 #set the confidence interval with the alpha_conf argument specified at the top
 ps    = pred.summary_frame(alpha=alpha_conf)
 pslo  = np.array(ps['mean_ci_lower'])
 pshi  = np.array(ps['mean_ci_upper'])
 npred = np.shape(pslo)[0]



 #plot the time series 9can also use dweek.plot() for simple option but less customisation
 x = dweek.axes[0]
 y = dweek['sum']
 
 if (combine == 0):
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
 else:
  ax1 = fig.add_subplot(np.int(np.ceil(nf/2.)),2,i+1)

 ax1.plot(x,y,label='data')
 
 xmod = ps.axes[0]
 ymean = np.array(ps['mean'])
 ylo  = np.array(ps['mean_ci_lower'])
 yhi  = np.array(ps['mean_ci_upper'])
 ax1.plot(xmod,ymean,label='forecast')
 ax1.set_title(lab)
 ax1.fill_between(xmod,ylo,yhi,alpha = 0.3,label='uncertainty')
 ax1.set_xlabel('Date')
 ax1.set_ylabel('Expense GBP')
 plt.legend(fontsize='xx-small')
 
 if (combine == 0):
  plt.savefig('fig_'+lab+'.png')



if (combine == 1):
 plt.tight_layout()
 plt.savefig('summary.png') 
 