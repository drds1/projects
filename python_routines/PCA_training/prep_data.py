#prepare a scatter plot of data and define custom multi-layer perceptron in
#another script to classify the points

from sklearn.datasets import make_blobs
import matplotlib.pylab as plt
import numpy as np

ndim = 3

#cov = np.ones((ndim,ndim))
#mean = np.zeros(ndim)

cov = np.diag(np.ones(ndim))
mean = np.zeros(ndim)


#off diagonal terms cause a rotation
cov[0,1] = 1.0
cov[2,0] = 1.0

cov[0,0] = 4.0

#random data multivariate gaussian
a = np.random.multivariate_normal(mean,cov,50000)




#plot data as test
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(a[:,0],a[:,1])
ax1.set_xlim([-5,5])
ax1.set_ylim([-5,5])

plt.show()


np.savetxt('pca_fake.dat',a)








#ny = 2
##generate samples and make blobs around 3 centroids
#X, y = make_blobs(n_samples=1000, n_features=ndim, centers=ny, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
#




#scattering of points clustered around 2 centroids. Need to use neural net to 
#predict which class a new point belongs.