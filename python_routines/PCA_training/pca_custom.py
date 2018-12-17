import numpy as np
import matplotlib.pylab as plt
import os



#load the data columns show each dimension ndim
#row show a given example of the input training data ndata
#dat = np.loadtxt('pca_fake.dat')
#
#
#
#
#
#
#
#
#
##set k. The number of eigen vectors to consider (MUST be less than Ndim)
#k = 3
##for the plot can give names to each of the axis labels. If empty list then default label
##is dimension 1, 2, 3 etc
#label=[]


def mypca(datin,k,diagfolder='',label=[],meannorm = 1,stdnorm=0): 
 
 if (type(datin) == np.ndarray):
  dat = 1.*datin
 elif (type(a) == str):
  dat = np.loadtxt(datin)
 else:
  raise Exception('Please enter input data datin as a numpy array or a file name containing data')
 
 ndat,ndim = np.shape(dat)
 
 if (k > ndim):
  raise Exception('Must use k < ndim to reduce the dimensionality of the parameter space')
 
 
 
 # do the bit on the data subtract mean and calculate eigen values and eigen vectors.
 # compute the covariance matrix, eigen vectors and eigen values
 if (meannorm == 1):
  mean = np.mean(dat,axis=0)
 else:
  mean = np.mean(dat,axis=0)*0
  
 if (stdnorm == 1):
  std = np.std(dat,axis=0)
 else:
  std = np.std(dat,axis=0)*0 + 1
 
 #subtract the mean
 submean = (dat - mean)/std
 cov  = np.cov(submean.T)
 #compute the eigen values and eigen vectors
 ev = np.linalg.eig(cov)
 eval = ev[0]
 evec = ev[1]
 #generate unit eigen vectors
 uevec = evec/np.sqrt(np.sum(evec**2,axis=1))
 #The proportion of the variance that each eigenvector represents can be 
 #calculated by dividing the eigenvalue corresponding to that eigenvector 
 #by the sum of all eigenvalues.
 evalsum = np.sum(eval)
 fraceval = eval/evalsum
 
 
 
 
 
 
 
 
 #print out the eigen vectors in order of fractional eigen values (large to small)
 idsort = np.argsort(fraceval)[-1::-1]
 for i in range(k):
  idx = idsort[i]
  print 'Eigen vector ',i+1,': ',uevec[:,idx],'.    Fractional variance: ',fraceval[idx]
 
 
 #diagnostic plot only for 2 and 3d
 if (diagfolder != ''):
  os.system('rm -rf '+diagfolder)
  os.system('mkdir '+diagfolder)
  for id1 in range(ndim):
   for id2 in range(ndim):
    if (id1 == id2):
     continue
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(dat[:,id1],dat[:,id2],s=2,color='k',label=None)
    
    xlim = list(ax1.get_xlim())
    xrange = xlim[1]-xlim[0]
    ylim = list(ax1.get_ylim())
    yrange = ylim[1]-ylim[0]
    
    maxrange = np.max([xrange,yrange])
    
    ax1.set_xlim([mean[0]-maxrange,mean[0]+maxrange])
    ax1.set_ylim([mean[1]-maxrange,mean[1]+maxrange])
    if (label == []):
     ax1.set_xlabel('dimension '+np.str(id1))
     ax1.set_ylabel('dimension '+np.str(id2))
    else:
     ax1.set_xlabel(label[id1])
     ax1.set_ylabel(label[id2])
  
    
    #ax1.set_ylim([-plotlim,plotlim])
    #get a line for each eigen vector
    for idx in range(k):
     i = idsort[idx]
     grad  = evec[1,i]/evec[0,i]
     yplot = mean[id2] + grad *(xlim - mean[id1])
     ax1.plot(xlim,yplot,ls='--',label='fractional EV '+np.str(idx+1)+'= '+np.str(np.round(fraceval[i],2)))
     plt.legend()
    #ax1.quiver([0,0],[0,0],[0+evec[0,0],0+evec[0,1]],[0+evec[1,0],0+evec[1,1]])
    plt.savefig(diagfolder+'/fig_'+np.str(id1)+'_'+np.str(id2)+'.pdf')
  
 
  
 #form the data in the k highest prnicpal component frame take transpose at the end to restore
 #the dat[nsamples,ndim] format of the input data
 newdat = np.matmul(evec[:,:].T,submean.T).T
 newdatsort = np.matmul(evec[:,idsort].T,submean.T).T
 
 #plot the data in the principal component space plotting the first principal component on
 #the x axis with y axis of subsequent plots showing the variance across the decresing 
 #eigen vectors 
 if (diagfolder != ''):
  for i in range(1,k):
   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   ax1.scatter(newdatsort[:,0],newdatsort[i],s=2)
   ax1.set_xlabel('EV 1')
   ax1.set_ylabel('EV '+np.str(i+1))
   plt.savefig(diagfolder+'/PCA_'+np.str(i)+'.pdf')
 
 
 return(evec,eval,fraceval,idsort,newdat,newdatsort,mean,std)



#input new definition to make new transformations on new data for eigen values
#already computed
def PCA_convert(dat,evec,meanin=0,stdin=0):
 
 if (meanin > 0):
  mean = 1.*meanin
 elif (meanin == 0):
  mean = 0
 else:
  mean = np.mean(dat,axis=0)
  
 if (stdin > 0):
  std = 1.*stdin
 elif (stdin == 0):
  std = 0
 else:
  std = np.std(dat,axis=0)

 
 
 ndat,ndim = np.shape(dat)
 datsub = np.zeros((ndat,ndim))
 for i in range(ndim):
  datsub[:,i] = (dat[:,i] - mean[i])/std[i]
 newdat = np.matmul(evec[:,:].T,datsub.T).T
 return(newdat) 


