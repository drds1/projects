import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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
 ax1.hist(x[ilo:ihi,0],histtype='step')
 ax1.plot(xres,yres)


plt.savefig('gmm_test.pdf')

print(gmm.means_)
print('\n')
print(gmm.covariances_)