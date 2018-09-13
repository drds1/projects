print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)
nepoch_train = 10000
nepoch_test  = 10000
nepoch_bad   = 1000
bad_in_train = 0
nts = 1
std = [0.3,0.3]
mean = [-2.0,2.0]
relativepeak = [1,1]

stdbad  = [0.2]
meanbad = [0.0]

separation = 2


nmode = len(std)
nmode_bad = len(stdbad)
#generate multimodal training data
X_train = []
rps = np.sum(relativepeak)
for i in range(nmode):
 nnow = np.int(1.*relativepeak[i]/rps*nepoch_train)
 X_train.append( rng.randn(nnow)*std[i] + mean[i] )
X_train = np.concatenate(np.array(X_train))


#generate the multimodal bad data following a different distribution
#the training will include the bad data

if (bad_in_train == 1):
 X_train_bad = []
 rps = np.sum(relativepeak)
 for i in range(nmode_bad):
  nnow = np.int(1.*relativepeak[i]/rps*nepoch_bad)
  X_train_bad.append( rng.randn(nnow)*stdbad[i] + meanbad[i] )
 X_train_bad = np.concatenate(np.array(X_train_bad))


#generate multimodal test_data following the same distribution
X_test = []
rps = np.sum(relativepeak)
for i in range(nmode):
 nnow = np.int(1.*relativepeak[i]/rps*nepoch_test)
 X_test.append( rng.randn(nnow)*std[i] + mean[i] )
X_test = np.concatenate(np.array(X_test))

#generate the multimodal bad data following a different distribution
#incorporate this into the test data
X_test_bad = []
rps = np.sum(relativepeak)
for i in range(nmode_bad):
 nnow = np.int(1.*relativepeak[i]/rps*nepoch_bad)
 X_test_bad.append( rng.randn(nnow)*stdbad[i] + meanbad[i] )
X_test_bad = np.concatenate(np.array(X_test_bad))


if (bad_in_train == 1):
 xtrain_tot = np.append(X_train,X_train_bad)
else:
 xtrain_tot = 1.*X_train
xtest_tot  = np.append(X_test,X_test_bad)
# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(xtrain_tot.reshape(-1,1))
y_pred_train = clf.predict(xtrain_tot.reshape(-1,1))
y_pred_test = clf.predict(X_test.reshape(-1,1))
y_pred_outliers = clf.predict(X_test_bad.reshape(-1,1))




xplot = xtest_tot
yplot = np.append(y_pred_test,y_pred_outliers)
nxp = np.shape(yplot)[0]
idx = np.arange(nxp)#np.random.choice(np.arange(nxp), size=nxp, replace=False, p=None)
xplot = xplot[idx]
yplot = yplot[idx]
idgood = np.where(yplot == 1)[0]
idbad  = np.where(yplot == -1)[0]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(idbad,xplot[idbad],ls='',marker='o',color='r',label='Time series (outliers)')
ax1.plot(xtrain_tot,label='training data')
#ax1.plot(idgood,xplot[idgood],label='Time series')
plt.legend()
plt.savefig('timeseries_isolationforrest.pdf')
plt.clf()


#make a histogram of the rejections
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(X_train,histtype='step',label='training sample (good)',bins=50,normed=True)
if (bad_in_train == 1):
 ax1.hist(X_train_bad,histtype='step',label='training sample (outliers)',bins=50,normed=True)
ax1.hist(X_test,histtype='step',label='test sample (good)',bins=50,normed=True)
ax1.hist(X_test_bad,histtype='step',label='test sample (true outliers)',bins=50,normed=True)
ax1.hist(xplot[idbad],histtype='step',label='test sample (identified outliers)',bins=50,normed=True)
plt.legend()
plt.savefig('histogram_isolationforrest.pdf')
plt.clf()

#zdec = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])