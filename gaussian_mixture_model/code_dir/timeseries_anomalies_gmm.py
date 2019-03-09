import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#based on this a little http://www.astroml.org/book_figures/chapter4/fig_GMM_1D.html
#but mainly my own training data and trial and error teaching


#function to use aic or bic to identify optimum number of components in GMM
def gmm_optimize(xin,comps = np.arange(1,10,1),stat='aic'):
 aic = []
 bic = []
 
 ncomps = np.shape(comps)[0]
 models = []
 for i in range(ncomps):
  cnow = comps[i]
  gmm = GaussianMixture(n_components=cnow)
  mnow=gmm.fit(xin)
  models.append(mnow)
  gmm.fit(xin)
  aic.append(mnow.aic(xin))
  bic.append(mnow.bic(xin))
  

 if (stat == 'aic'):
  ibest = np.argmin(aic)
 elif (stat == 'bic'):
  ibest = np.argmin(bic)
 
 print 'optimum number of gmm components',comps[ibest]
 print 'aic',aic[ibest]
 print 'bic',bic[ibest]

 labels = models[ibest].predict(xin)
 output = {'gmm optimum model':models[ibest],
           'labels':labels,
           'means':models[ibest].means_,
           'covariances':models[ibest].covariances_
           }
 return(output)












if __name__ == "__main__":
 #code to test function and make fancy plot

 nsamp = [100,100,100]
 means = [1.,10.,15.]
 sd    = [2.,2.,2.]



 #generate fake data
 ntot = np.sum(nsamp)
 ndis = len(nsamp)
 x = np.zeros((0,2))
 for i in range(ndis):
  a = np.random.randn(nsamp[i])*sd[i] + means[i]
  b = np.zeros(nsamp[i]) + i
  x = np.vstack((x,np.array([a,b]).T))




 #train GMM
 xin = np.reshape(x[:,0],(ntot,-1))
 gmm = GaussianMixture(n_components=3)
 gmm.fit(xin)

 xres = np.arange(-10,50,0.1)

 nres = np.shape(xres)[0]
 xr   = np.reshape(xres,(nres,-1))
 yres = np.exp(gmm.score_samples(xr))




 fig = plt.figure()
 ax1=fig.add_subplot(111)
 ilo = 0
 for i in range(ndis):
  if (i > 0):
   ilo = ihi
  ihi = ilo + nsamp[i]
  ax1.hist(x[ilo:ihi,0],histtype='step',normed=True)
  ax1.plot(xres,yres)


 plt.savefig('gmm_test.pdf')

 print(gmm.means_)
 print('\n')
 print(gmm.covariances_)




 #now repeat but use AIC and BIC to identify optimum number of components
 comp = np.arange(1,9,1)
 ncomp = np.shape(comp)[0]



 aic = []
 bic = []
 axes = []
 yres = []
 for i in range(ncomp):
  cnow = comp[i]
  gmm = GaussianMixture(n_components=cnow)
  gmm.fit(xin)
  aic.append(gmm.aic(xin))
  bic.append(gmm.bic(xin))
  yres.append(np.exp(gmm.score_samples(xr)))
  print 'ncomponents...',cnow,'    aic',aic[i],'     bic',bic[i]




 #make plots of all the tested number of components
 fig = plt.figure()
 i_best = np.argmin(aic)
 for i in range(ncomp):
  cnow = comp[i]
  ax1 = fig.add_subplot(np.int(np.ceil(ncomp/2.)),2,i+1)
  axes.append(ax1)
  ilo = 0
  for i2 in range(ndis):
   if (i2 > 0):
    ilo = ihi
   ihi = ilo + nsamp[i2]
   #yres = np.exp(gmm.score_samples(xr))
   ax1.hist(x[ilo:ihi,0],histtype='step',normed=True)
  ax1.plot(xres,yres[i])
  ax1.text(0.99,0.3,'n='+np.str(cnow)+'\naic='+np.str(np.int(aic[i]))+'\nbic='+np.str(np.int(bic[i])),ha='right',transform=ax1.transAxes)
  if (i == i_best):
   axes[i_best].text(0.0,1.05,'Best fit (min AIC)',ha='left',transform=ax1.transAxes,color='b',fontweight='bold')

 plt.tight_layout()
 plt.savefig('fig_aicbic_comp.pdf')


 print 'using function'
 a=gmm_optimize(xin)








 
 
 