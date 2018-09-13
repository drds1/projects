print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

model = 'isolation forrest'#'local outlier fraction'

rng = np.random.RandomState(42)
nepoch_train = 100
nepoch_test  = 100
nbad   = 20
nts = 2
std = 0.3
separation = 2


# Generate train data
X_train = std * rng.randn(nepoch_train, nts)
for i in range(nts):
 X_train[:,i] += i*separation 
 
 
# Generate some regular novel observations following the same pattern
X_test = std * rng.randn(nepoch_test, nts)
for i in range(nts):
 X_test[:,i] += i*separation 
 
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(nbad, nts))

# fit the model
x_comb = np.vstack((X_train,X_test,X_outliers))
if (model == 'isolation forrest'):
 clf = IsolationForest(max_samples=100, random_state=rng)
 clf.fit(X_train)
 y_pred_train = clf.predict(X_train)
 y_pred_test = clf.predict(X_test)
 y_pred_outliers = clf.predict(X_outliers)
 y_pred = clf.predict(x_comb)
elif (model == 'local outlier fraction'):
 clf = LocalOutlierFactor(n_neighbors=20)
 y_pred = clf.fit_predict(x_comb)
 
 
 
# plot the line, the samples, and the nearest vectors to the plane
if (nts > 1):
 
 if (model == 'isolation forrest'):
  xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
  Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  
  plt.title("Isolation Forest")
  plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
  
  b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                   s=20, edgecolor='k')
  b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                   s=20, edgecolor='k')
  c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                  s=20, edgecolor='k')
  id_flag = np.where(y_pred == -1)[0]
  d = plt.scatter(x_comb[id_flag, 0], x_comb[id_flag, 1], c='red',
                  edgecolor='k', s=70,marker='x',zorder=1)
  plt.axis('tight')
  plt.xlim((-5, 5))
  plt.ylim((-5, 5))
  plt.legend([b1, b2, c,d],
             ["training observations",
              "new regular observations", "new abnormal observations", "outliers flagged"],
             loc="upper left")
  plt.savefig('isolation_forrest.pdf')
 elif (model == 'local outlier fraction'):
  # plot the level sets of the decision function
  xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
  Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  
  plt.title("Local Outlier Factor (LOF)")
  plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
  
  a = plt.scatter(x_comb[:nepoch_train+nepoch_test, 0], x_comb[:nepoch_train+nepoch_test, 1], c='white',
                  edgecolor='k', s=20,zorder=2)
  b = plt.scatter(x_comb[nepoch_train+nepoch_test:, 0], x_comb[nepoch_train+nepoch_test:, 1], c='red',
                  edgecolor='k', s=20,zorder=3)
  
  id_flag = np.where(y_pred == -1)[0]
  c = plt.scatter(x_comb[id_flag, 0], x_comb[id_flag, 1], c='red',
                  edgecolor='k', s=70,marker='x',zorder=1)
  plt.axis('tight')
  plt.xlim((-5, 5))
  plt.ylim((-5, 5))
  plt.legend([a, b,c],
             ["normal observations",
              "outlier observations",
              "outliers identified"],
             loc="upper left")
  plt.show()


else:
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 
 plt.savefig('isolation_forrest.pdf')

plt.clf()


yplot = 1.*y_pred
nxp = np.shape(yplot)[0]
idx = np.random.choice(np.arange(nxp), size=nxp, replace=False, p=None)
yplot = yplot[idx]
#now plot the time series showing the outlying data
fig = plt.figure()
for i in range(nts):
 xplot = x_comb[idx,i]#np.append(X_test[:,i],X_outliers[:,i])[idx]
 ax1 = fig.add_subplot(nts,1,i+1)
 idgood = np.where(yplot == 1)[0]
 idbad  = np.where(yplot == -1)[0]
 ax1.plot(xplot,label='light curve')
 #ax1.plot(idgood,xplot[idgood],label='light curve')
 ax1.scatter(idbad,xplot[idbad],color='r',label='flagged outliers')
 id = idx[nepoch_train+nepoch_test:]
 ax1.scatter(id,xplot[id],color='r',marker='x',s=50,label='True outliers')

nepoch_train = 100
nepoch_test  = 100
nbad   = 20
plt.legend()
plt.savefig('isolation_forrest_timeseries.pdf')
