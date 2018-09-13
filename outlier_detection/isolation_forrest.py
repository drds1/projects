print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)
nepoch = 100
nbad   = 20
nts = 1
std = 0.3
separation = 2


# Generate train data
X_train = std * rng.randn(nepoch, nts)
for i in range(nts):
 X_train[:,i] += i*separation 
 
 
# Generate some regular novel observations following the same pattern
X_test = std * rng.randn(nepoch, nts)
for i in range(nts):
 X_test[:,i] += i*separation 
 
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(nbad, nts))

# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
if (nts > 1):
 xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
 Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
 Z = Z.reshape(xx.shape)
 
 plt.title("IsolationForest")
 plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
 
 b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                  s=20, edgecolor='k')
 b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                  s=20, edgecolor='k')
 c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                 s=20, edgecolor='k')
 plt.axis('tight')
 plt.xlim((-5, 5))
 plt.ylim((-5, 5))
 plt.legend([b1, b2, c],
            ["training observations",
             "new regular observations", "new abnormal observations"],
            loc="upper left")
 plt.savefig('isolation_forrest.pdf')
else:
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 
 plt.savefig('isolation_forrest.pdf')





yplot = np.append(y_pred_test,y_pred_outliers)
nxp = np.shape(yplot)[0]


idx = np.random.choice(np.arange(nxp), size=nxp, replace=False, p=None)
yplot = yplot[idx]
#now plot the time series showing the outlying data
fig = plt.figure()
for i in range(nts):
 xplot = np.append(X_test[:,i],X_outliers[:,i])[idx]
 ax1 = fig.add_subplot(nts,1,i+1)
 idgood = np.where(yplot == 1)[0]
 idbad  = np.where(yplot == -1)[0]
 ax1.plot(idgood,xplot[idgood],label='light curve')
 ax1.scatter(idbad,xplot[idbad],color='r',label='Outliers')

plt.savefig('isolation_forrest_timeseries.pdf')
