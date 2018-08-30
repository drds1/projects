import os
os.system('rm -rf code_pythonsubs.py')
os.system('rm -rf code_pythonsubs.pyc')
os.system('python prep_codes.py')
import numpy as np
import code_pythonsubs as hdp
import matplotlib.pylab as plt



#load the input heart data from the cleavland heart disease 90's data
#source 
#1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
#2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
#3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
#4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D. 

#classification problem correlating heart disease with 14 measurable attributes/parameters/dimensions
#1. #3 (age)
#2. #4 (sex)
#3. #9 (cp)
#4. #10 (trestbps)
#5. #12 (chol)
#6. #16 (fbs)
#7. #19 (restecg)
#8. #32 (thalach)
#9. #38 (exang)
#10. #40 (oldpeak)
#11. #41 (slope)
#12. #44 (ca)
#13. #51 (thal)
#14. #58 (num) (the predicted attribute) 

dat = np.genfromtxt('input_data.dat',delimiter=',')
#remove records with nans
dat = dat[~np.isnan(dat).any(axis=1)]
#remove the classification from the data
clas = dat[:,-1]
#dat  = np.delete(dat,-1,1)

diagplots = 1

#dat = np.loadtxt('make_fake_dim.dat')
#clas = np.loadtxt('make_fake_class.dat')




nsamp,ndim = np.shape(dat)



ntest = 10
#extract a subsample of data points for training
id  = np.random.randint(0,nsamp,ntest)
d_test = dat[id,:]
c_test_true = clas[id]

d_train = np.delete(dat,id,0)
c_train = np.delete(clas,id)


print 'extracting subsample of points for training and classification test'
for i in range(ntest):
 print id[i],d_test[i,:]
 

print ''
print 'fitting k nearest neighbour'
#perform a k nearest neighbour algorithm on the test data to predict which level of heart
#disease will be present in a test patient
c_test = hdp.k_test(d_test,d_train,c_train,distance=2,k=3)



print 'comparing the test with true classifications'
for i in range(ntest):
 print 'test class: true class ',c_test[i],c_test_true[i]

idmatch = np.where(c_test == c_test_true)[0]
nmatch = np.shape(idmatch)[0] 
print 'accuracy...', 1.*nmatch/ntest





#plot the individual pairs of dimensions in the parameter space to investigate the correlations
if (diagplots == 1):
 os.system('rm -rf diagplots')
 os.system('mkdir diagplots')
 idclas = np.unique(clas)
 nclass = np.shape(idclas)[0]
 for i in range(ndim):
  for i2 in range(ndim):
   if (i == i2):
    continue
   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   for i3 in range(nclass):
    idcnow = idclas[i3]
    idplot = np.where(clas == idcnow)[0]
    ax1.scatter(dat[idplot,i],dat[idplot,i2],s=2,label='class '+np.str(idcnow))
   plt.legend(fontsize='x-small')
   plt.savefig('./diagplots/fig_'+np.str(i)+'_'+np.str(i2)+'.pdf') 
   

#perform a PCA analysis to understand the correlations between each dimensions and
#find out along which axis lie the maximum variance
print 'performing pca...'
k = ndim
evec,eval,fraceval,idsort,datnew,datnewsort = hdp.mypca(dat,k,diagfolder='',label=[]) 

if (diagplots == 1):
 print 'plotting variance along principal component axis'
 for i in range(1,k):
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  for i3 in range(nclass):
   idcnow = idclas[i3]
   idplot = np.where(clas == idcnow)[0]
   ax1.scatter(datnewsort[idplot,0],datnewsort[idplot,i],s=2,label='class '+np.str(idcnow))
  ax1.set_xlabel('EV 1')
  ax1.set_ylabel('EV '+np.str(i+1))
  plt.legend(fontsize='x-small')
  plt.savefig('./diagplots/fig_PCA_'+np.str(i)+'.pdf')

for ik in range(k):
 id = idsort[ik]
 print 'eigen vector ',ik+1,' frac var=',fraceval[id],'variance along pca axis',np.std(datnew[:,ik]),np.std(datnewsort[:,ik])


#nk_use = 3
##isolate the top 3 principle components and plot the classifications along those axis
#evec_new = evec[:,idsort[:k]]
#eval_new = eval[idsort[:k]]
#fraceval_new = fraceval[idsort[:k]]
#dat_pca  = dat* 
#






