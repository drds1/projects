import numpy as np
import my_kateccf as mccf
import mylcgen as mlc
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt


ilag = 5
sigma = 0.7
plotlim = [-50,50]
plotlim_peak = [-50,50]
#generate synthetic data
drive = mlc.mylcgen(datfile='',p0=1.0,f0=0.1,a=-2,b=-2,tlo=0,thi=100,dt=1.0,ploton=0,iseed=1234,meannorm = -1., sdnorm = -1.0)
 
t = drive[:,0]
x = drive[:,1]
xsd = np.std(x)
nx = np.shape(x)[0] 
x = (x - np.mean(x))/xsd
xsd = np.std(x)
sigx = np.ones(nx)*xsd*sigma
x = x + np.random.randn(nx)*sigx

lc1 = np.array([t,x,sigx]).T

#define convolution kernel
conv = np.zeros(nx)
conv[ilag-1:ilag+1] = 1



echo  = np.convolve(x, conv, mode='same')
echo2 = np.convolve(x, conv, mode='full')[:nx]
 
echo2 = (echo2 - np.mean(echo2))/np.std(echo2) 
esd = np.std(echo2)
ne2=np.shape(echo2)[0]
sige = np.zeros(ne2)+esd*sigma
echo2 = echo2 + np.random.randn(ne2)*esd*sigma
lc2 = np.array([t,echo2,sige]).T

#plot results
ndown = 10
nalong = 5
gs1 = gridspec.GridSpec(ndown,nalong)

gs1.update(left=0.15,right=0.85,bottom=0.05,top=0.99,wspace=0.1,hspace=0.1)

ax1 = plt.subplot(gs1[:3,:])
#ax1.plot(t,x)
ax1.errorbar(t,x,sigx,label='Time series 1 (e.g arbitrage)')
xl = list(ax1.get_xlim())
ax1.plot(xl,[0,0],label=None,ls='--',color='k')

ax3 = plt.subplot(gs1[4:7,:])
#ax3.plot(t,echo2)
ax3.errorbar(t,echo2,sige,label='Time series 2 (e.g flow)')
xl = list(ax3.get_xlim())
ax3.plot(xl,[0,0],label=None,ls='--',color='k')

ax2 = plt.subplot(gs1[8:,0])
ax2.plot(t,conv)
ax2.set_xlim([ilag-5,ilag+5])
ax2.set_ylabel('Convolution kernel')
ax2.set_xlabel('lag (days)')




#compute ccf function
print 'scipy ccf'
import scipy.signal as ss
ccf = ss.correlate(x,echo2)
print 'done'

import statsmodels.tsa.stattools as st
print 'statsm ccf'
ccf = st.ccf(x[ilag+2:],echo2[ilag+2:])
print 'done'
nccf = np.shape(ccf)[0]
tccf = np.arange(nccf) - nccf/2
ax4 = plt.subplot(gs1[8:,2])
ax4.plot(tccf,ccf)
yl = list(ax4.get_ylim())
ax4.plot([0,0],yl,color='k',ls='--',label='ccf function')
ax4.set_ylim(yl)
ax4.set_ylabel('CCF(lag)')
ax4.set_xlabel('lag (days)')
ax4.set_xlim([-ilag-5,ilag+5])



#compute looped distribution of mean lags

lagrange = [-50,50]
print 'kate ccf'
a = mccf.kateccf(lc1[ilag+2:,:],lc2[ilag+2:,:],interp = 0.2, nsim = 500, mcmode = 1, sigmode = -2.e26,
 plotop = '', fileop = 0,output_file_folder = './',filecent='sample_centtab.dat',
 filepeak='sample_peaktab.dat',filesample='sample_peaktab.dat')
rmax = np.array(a[-1])
#tlags_centroid,tlags_peak,lag,r,centau,centau_uperr,centau_loerr,peaktau,peaktau_uperr,peaktau_loerr 

#tlags_centroid,tlags_peak,lag,r,centau,centau_uperr,centau_loerr,peaktau,peaktau_uperr,peaktau_loerr,
 #nsuccess_peak, nfail_peak, nsuccess_centroid, nfail_centroid, max_rvals, nfail_rvals, pvals 
#lag_range=[-100,100]
print 'done'
lag_centroid_dist = a[0]
idx = np.where((lag_centroid_dist > lagrange[0]) & (lag_centroid_dist < lagrange[1]))[0]
lag_centroid_dist = lag_centroid_dist[idx]
rmax = rmax[idx]
lo,med,hi = np.percentile(lag_centroid_dist,[15.865,50,84.135])#a[4,5,6]
ax5 = plt.subplot(gs1[8:,4])
ax5.hist(lag_centroid_dist,bins=50,alpha=1.0,weights=rmax,normed=True,label='lag distribution')
ax5.set_xlabel('lag (days)')
ax5.set_ylabel('iterated \n mean lag \n distribution')
yl = list(ax5.get_ylim())
ax5.plot([lo]*2,yl,color='k',ls='--')
ax5.plot([hi]*2,yl,color='k',ls='--')
ax5.plot([med]*2,yl,color='k',ls='-')
ax5.set_xlim(plotlim)

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()




plt.savefig('fig_example_timeseries.pdf')


lag_peak_dist = a[1]
idx = np.where((lag_peak_dist > lagrange[0]) & (lag_peak_dist < lagrange[1]))
lag_peak_dist = lag_peak_dist[idx]
lo,med,hi = np.percentile(lag_peak_dist,[15.865,50,84.135])#a[4,5,6]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(lag_peak_dist,bins=50,alpha=1.0,label='lag distribution')
yl = list(ax1.get_ylim())
ax1.plot([lo]*2,yl,color='k',ls='--')
ax1.plot([hi]*2,yl,color='k',ls='--')
ax1.plot([med]*2,yl,color='k',ls='-')
ax1.set_xlabel('peak distribution lag (days)')
ax1.set_ylabel('iterated \n mean lag \n distribution')
ax1.set_xlim(plotlim_peak)
plt.savefig('kate_ccf_peak_hist.pdf')

#ccf.kateccf(lc1,lc2,lag_range=[-100,100],interp = 2.0, nsim = 500, mcmode = 0, sigmode = 0.2, plotop = '', fileop = 0,output_file_folder = './',filecent='sample_centtab.dat',filepeak='sample_peaktab.dat',filesample='sample_peaktab.dat'):
