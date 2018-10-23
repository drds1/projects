import statsmodels.api as sm
import numpy as np


#code to trial several starting parameters for the tsa.statespace SARIMA model
# and identify the best combination of input parameters (minimising aic)
def aic_test(dfin,order=[5,5,5]):
 na,nb,nc = order
 opsave = np.zeros((0,4))
 for ia in range(na):
  for ib in range(nb):
   for ic in range(nc):
    try:
     order = (ia,ib,ic)
 
     #use data science to forecast future variability
     model=sm.tsa.statespace.SARIMAX(endog=dfin,order=order,seasonal_order=(0,1,0,12),trend='c',enforce_invertibility=False)
     results = model.fit()
     aic = results.aic
     
     opsave = np.vstack((opsave,[aic,ia,ib,ic]))
    except:
     print 'wrong combination of ia,ib,ic',ia,ib,ic
    
 idx_sort = np.argsort(opsave[:,0])
 op_output = opsave[idx_sort,:]
 return(op_output)
 
 
 


#the aic test sometimes doesnt work
#use cross validation by splitting the data set into a training and test data
#for each combination of parameters, compute the chi squared of the 'forecast'
#where the forecast is just the data excluded from the training data by the cross 
#

def cv_test(dfin,order=[5,5,5],fraction_cv = 0.8):
 na,nb,nc = order
 opsave = np.zeros((0,4))
 ndat   = dfin.size
 ilo    = np.int(fraction_cv*ndat)
 train  = dfin[:ilo]
 test   = dfin[ilo:]
 for ia in range(na):
  for ib in range(nb):
   for ic in range(nc):
    try:
     order = (ia,ib,ic)
     #use sarima model to forecast future variability
     model   = sm.tsa.statespace.SARIMAX(endog=train,order=order,seasonal_order=(0,1,0,12),trend='c',enforce_invertibility=False)
     results = model.fit()
     pred  = results.get_prediction(start = ilo, end = ndat )
     psmod    = pred.summary_frame(alpha=0.05)['mean'].values
     cisq  = (pred - test)**2
     opsave = np.vstack((opsave,[cisq,ia,ib,ic]))
    except:
     print 'wrong combination of ia,ib,ic',ia,ib,ic
 idx_sort = np.argsort(opsave[:,0])
 op_output = opsave[idx_sort,:]     
 return()     
      