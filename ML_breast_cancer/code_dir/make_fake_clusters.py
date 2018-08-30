#prepare a scatter plot of data and define custom multi-layer perceptron in
#another script to classify the points

from sklearn.datasets import make_blobs
import numpy as np

ndim = 2
ny = 3
#generate samples and make blobs around 3 centroids

print 'generating fake samples'
X, y = make_blobs(n_samples=1000, n_features=ndim, centers=ny, cluster_std=0.1, center_box=(-10.0, 10.0), shuffle=True, random_state=None)





#scattering of points clustered around 2 centroids. Need to use neural net to 
#predict which class a new point belongs.

#save data
np.savetxt('make_fake_dim.dat',X)
np.savetxt('make_fake_class.dat',y)





import matplotlib.pylab as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)

for i in range(ny):
 idx = np.where(y == i)[0]
 ax1.scatter(X[idx,0],X[idx,1])
plt.savefig('fake_data_class.pdf')