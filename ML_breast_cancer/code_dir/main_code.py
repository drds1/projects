import os
os.system('rm -rf code_pythonsubs.py')
os.system('rm -rf code_pythonsubs.pyc')
os.system('python prep_codes.py')
import numpy as np
import code_pythonsubs as hdp
import matplotlib.pylab as plt
import time


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





#train the machine learning algorithm on the original data




dat = np.genfromtxt('input_data.dat',delimiter=',')
#remove records with nans
dat = dat[~np.isnan(dat).any(axis=1)]
#remove the classification from the data
clas = dat[:,-1]
diagplots = 0
nsamp,ndim = np.shape(dat)



ntest = 10
l_rate=0.5
niterations=500
n_hidden = 5
#extract a subsample of data points for training
idtest  = np.random.randint(0,nsamp,ntest)
d_test = dat[idtest,:]
c_test_true = clas[idtest]

d_train = np.delete(dat,idtest,0)
c_train = np.delete(clas,idtest)


#now examine the loss of accuracy as you reduce the training sample for both neural net and knn
ntrain = np.shape(d_train[:,0])[0]
nexperiment = 10
ntsamp = np.linspace(0,ntrain,nexperiment+1,dtype='int')[1:]

naverage = 4

accuracy_nn = np.zeros((nexperiment,naverage))
accuracy_knn = np.zeros((nexperiment,naverage))
time_nn_train = np.zeros((nexperiment,naverage))
time_nn_test = np.zeros((nexperiment,naverage))
time_knn = np.zeros((nexperiment,naverage))
PCA_accuracy_nn = np.zeros((nexperiment,naverage))
PCA_accuracy_knn = np.zeros((nexperiment,naverage))
PCA_time_nn_train = np.zeros((nexperiment,naverage))
PCA_time_nn_test = np.zeros((nexperiment,naverage))
PCA_time_knn = np.zeros((nexperiment,naverage))

for iave in range(naverage):
 iexp = 0
 for sampsize in ntsamp:
  
  print 'sample size = ',sampsize
  idtrain = np.random.randint(0,ntrain,sampsize)
  print 'training neural network...'
  
  traintime0 = time.time()
  network,idkey = hdp.train_net(d_train[idtrain,:],c_train[idtrain],l_rate,niterations,n_hidden)
  traintime1 = time.time()
  testtime0 = time.time()
  opold,op = hdp.test_net(d_test,network,idkey=idkey)
  testtime1 = time.time()
  ncorrect = np.shape(np.where(op == c_test_true)[0])[0]
  acc_nn = 1.*ncorrect/ntest
  print 'accuracy of neural network =',acc_nn
  print 'training time = ',traintime1 - traintime0
  print 'test time = ',testtime1 - testtime0
  time_nn_train[iexp,iave] = traintime1-traintime0
  time_nn_test[iexp,iave]  = testtime1-testtime0
  
  accuracy_nn[iexp,iave] = acc_nn
  
  print 'running k nearest neighbour'
  knntime0 = time.time()
  op_knn = hdp.k_test(d_test,d_train[idtrain,:],c_train[idtrain],distance=2,k=3)
  knntime1 = time.time()
  print 'knn time = ',knntime1-knntime0
  ncorrect_knn = np.shape(np.where(op_knn == c_test_true)[0])[0]
  acc_knn = 1.*ncorrect_knn/ntest
  print 'accuracy of nearest neighbour =',acc_knn
  accuracy_knn[iexp,iave] = acc_knn
  time_knn[iexp,iave] = knntime1-knntime0
  
  
  
  
  #now employ PCA and repeat the experiment. Expect some loss of accuracy but perhaps not a huge amount
  print 'performing pca... using first two principal components'
  k = 2
  evec,eval,fraceval,idsort,datnew,datnewsort = hdp.mypca(d_train[idtrain,:],k,diagfolder='',label=[]) 
 
  traintime0 = time.time()
  network,idkey = hdp.train_net(datnewsort,c_train[idtrain],l_rate,niterations,n_hidden)
  traintime1 = time.time()
  d_test_pca = hdp.PCA_convert(d_test,evec)
  d_train_pca = 1.*datnew
  testtime0 = time.time()
  opold,op = hdp.test_net(d_test_pca,network,idkey=idkey)
  testtime1 = time.time()
  ncorrect = np.shape(np.where(op == c_test_true)[0])[0]
  acc_nn = 1.*ncorrect/ntest
  print 'accuracy of neural network =',acc_nn
  print 'training time = ',traintime1 - traintime0
  print 'test time = ',testtime1 - testtime0
  PCA_time_nn_train[iexp,iave] = traintime1-traintime0
  PCA_time_nn_test[iexp,iave] = testtime1-testtime0
  
  PCA_accuracy_nn[iexp,iave] = acc_nn
  
  print 'running k nearest neighbour'
  knntime0 = time.time()
  op_knn = hdp.k_test(d_test_pca,d_train_pca,c_train[idtrain],distance=2,k=3)
  knntime1 = time.time()
  print 'knn time = ',knntime1-knntime0
  ncorrect_knn = np.shape(np.where(op_knn == c_test_true)[0])[0]
  acc_knn = 1.*ncorrect_knn/ntest
  print 'accuracy of nearest neighbour =',acc_knn
  PCA_accuracy_knn[iexp,iave] = acc_knn
  PCA_time_knn[iexp,iave] = knntime1-knntime0
  iexp = iexp + 1
  
#make plots 
#scatter plot of classification space 2D in PCA space
fig = plt.figure()
ax1 = fig.add_subplot(111)
pca_dat = hdp.PCA_convert(dat,evec)
nclass = np.shape(idkey)[0]
col = ['b','r','k','orange','purple','cyan','magenta']*10
for iclass in range(nclass):
 idplot = np.where(clas == idkey[iclass])[0]
 ax1.scatter(pca_dat[idplot,0],pca_dat[idplot,1],c=col[iclass],label='class '+np.str(iclass+1))

plt.legend()
ax1.set_xlabel('Eigenvector 1')
ax1.set_ylabel('Eigenvector 2')
plt.savefig('PCA_ev12.pdf')



accuracy_knn = np.array(accuracy_knn)
PCA_accuracy_knn = np.array(PCA_accuracy_knn)
accuracy_nn = np.array(accuracy_nn)
PCA_accuracy_nn = np.array(PCA_accuracy_nn)

#figure comparing the computation time of neural net and k nearest neighbour
fig = plt.figure()
ax1 = fig.add_subplot(111)
y = np.mean(accuracy_knn,axis = 1)
sig = np.std(accuracy_knn,axis = 1)
ax1.errorbar(ntsamp,y*100,sig*100,marker='o',label='K-nearest-neighbour')

y = np.mean(PCA_accuracy_knn,axis=1)
sig = np.std(PCA_accuracy_knn,axis=1)
ax1.errorbar(ntsamp,y*100,sig*100,marker='o',label='K-nearest-neighbour (PCA)')


y = np.mean(accuracy_nn,axis=1)
sig = np.std(accuracy_nn,axis=1)
ax1.errorbar(ntsamp,y*100,sig*100,marker='o',label='neural network')

y = np.mean(PCA_accuracy_nn,axis=1)
sig = np.std(PCA_accuracy_nn,axis=1)
ax1.errorbar(ntsamp,y*100,sig*100,marker='o',label='neural network (PCA)')


ax1.set_xlabel('number of training examples')
ax1.set_ylabel('accuracy (%)')
plt.legend()
plt.savefig('fig_accuracy.pdf') 

#plot comparing computation time 
fig = plt.figure()
ax1 = fig.add_subplot(111)

y = np.mean(time_knn,axis=1)
sig = np.std(time_knn,axis=1)
ax1.errorbar(ntsamp,y,sig,marker='o',label='K-nearest-neighbour')

y = np.mean(PCA_time_knn,axis=1)
sig = np.std(PCA_time_knn,axis=1)
ax1.errorbar(ntsamp,y,sig,marker='o',label='K-nearest-neighbour (PCA)')


y = np.mean(time_nn_train,axis=1)
sig = np.std(time_nn_train,axis=1)
ax1.errorbar(ntsamp,y,sig,marker='o',label='Training neural network')

y = np.mean(time_nn_test,axis=1)
sig = np.std(time_nn_test,axis=1)
ax1.errorbar(ntsamp,y,sig,marker='o',label='Testing neural network')


y = np.mean(PCA_time_nn_train,axis=1)
sig = np.std(PCA_time_nn_train,axis=1)
ax1.errorbar(ntsamp,y,sig,marker='o',label='Training neural network (PCA)')

y = np.mean(PCA_time_nn_test,axis=1)
sig = np.std(PCA_time_nn_test,axis=1)
ax1.errorbar(ntsamp,y,sig,marker='o',label='Testing neural network (PCA)')

ax1.set_xlabel('number of training examples')
ax1.set_ylabel('computation time (seconds)')
ax1.set_yscale('log')
plt.legend()
plt.savefig('fig_computetime.pdf') 



