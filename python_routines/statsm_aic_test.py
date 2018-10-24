import statsmodels.api as sm
import numpy as np
import matplotlib.pylab as plt

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

def cv_test(dfin,order=[5,5,5],fraction_cv = 0.8,diagnostic_plot=0):
 na,nb,nc = order
 opsave = np.zeros((0,5))
 ndat   = dfin.size
 ilo    = np.int(fraction_cv*ndat)
 train  = dfin[:ilo]
 test   = dfin[ilo:]
 
 if (diagnostic_plot == 1):
  from matplotlib.backends.backend_pdf import PdfPages
  pdf = PdfPages('statsm_diagplot.pdf')
 
 for ia in range(na):
  for ib in range(nb):
   for ic in range(nc):
    try:
     order = (ia,ib,ic)
     #use sarima model to forecast future variability
     model   = sm.tsa.statespace.SARIMAX(endog=train,order=order,seasonal_order=(0,1,0,12),trend='c',enforce_invertibility=False)
     results = model.fit()
     pred  = results.get_prediction(start = ilo+1, end = ndat )
     psmod    = pred.summary_frame(alpha=0.05)['mean'].values
     
     
     cisq  = np.sum((psmod - test.values)**2)
     aic   = results.aic
     
     
     opsave = np.vstack((opsave,[cisq,aic,ia,ib,ic]))
     
     if (diagnostic_plot == 1):
      fig = plt.figure()
      ax1 = fig.add_subplot(111)
      ps_train = results.get_prediction(start = 0,end=ilo ).summary_frame(alpha=0.05)
      xmod_train = ps_train.axes[0]
      ymean      = np.array(ps_train['mean'])
      ylo        = np.array(ps_train['mean_ci_lower'])
      yhi        = np.array(ps_train['mean_ci_upper'])
      ax1.plot(xmod_train,ymean,label='model')
      ax1.plot(dfin,label='data')
 
      ps_test = results.get_prediction(start = ilo,end=ndat ).summary_frame(alpha=0.05)
      xmod_test = ps_test.axes[0]
      ymean      = np.array(ps_test['mean'])
      ylo        = np.array(ps_test['mean_ci_lower'])
      yhi        = np.array(ps_test['mean_ci_upper'])
      ax1.plot(xmod_test,ymean,label='forecast')
      ax1.fill_between(xmod_test,ylo,yhi,alpha=0.3,label='uncertainty')
      lab = r'$\chi^2='+np.str(cisq)+'$ $aic='+np.str(aic)+'$' +\
      '\n  a='+np.str(ia)+'  b='+np.str(ib)+'  c='+np.str(ic)
      ax1.annotate(lab, xy=(0.9, 0.9),  xycoords='axes fraction',
           xytext=(0.9, 0.9), textcoords='axes fraction',
           horizontalalignment='right', verticalalignment='top',
           )
      pdf.savefig(fig)
      print 'saving figure ...'
    except:
     print 'wrong combination of ia,ib,ic',ia,ib,ic
 
 if (diagnostic_plot == 1):
  pdf.close()
 idx_sort = np.argsort(opsave[:,0])
 op_output = opsave[idx_sort,:]     

 return(op_output)     
      