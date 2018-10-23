#python project to load in numbers spreadsheets for each 
#exp_xxxx.numbers file in /Users/david/projects/expenses_data

import numpy as np
import matplotlib.pylab as plt
import glob
import pandas as pd
import datetime

dir_input = '/Users/david/projects/expenses_data'
#load each file in the spreadsheet directory
files = glob.glob(dir_input + '/exp_*.csv')



nf = len(files)

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
 
 
 
 
 #plot the time series
 #fig = plt.figure()
 #ax1=fig.add_subplot(111)
 #ax1.set_title(lab)
 #ax1.plot(dweek)#dweek.plot()
 d = dweek.plot()
 d.set_title(lab)
 plt.show()
 
 