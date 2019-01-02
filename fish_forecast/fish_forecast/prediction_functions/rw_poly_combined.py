import numpy as np


#the polynomial and fourier fitting should be performed together rather than one after the other



def rw(ti,yi,si=None,tgi = -1,fbreak=-1,custom_freqs=[],nits = 1,order = 1):


 #initialise starting inputs and normalise
 twopi = 2*np.pi
 t = np.array(ti)
 tlo = np.min(t)
 thi = np.max(t)
 dt  = t[1] - t[0]
 ymean = np.mean(yi)
 ystd  = np.std(yi)
 y = (yi - ymean)/ystd
 ny = len(y)
 rms = np.std(y)

 if si is None:
  s = np.ones(ny)
 else:
  s = si/ystd

 if (type(tgi) == np.ndarray): 
  tgrid = np.array(tgi)
 else:
  tgrid = np.arange(tlo,thi+dt,dt)
 ntgrid = len(tgrid)
 dtm = tgrid[1]-tgrid[0]
 tglo,tghi = tgrid[0],tgrid[-1]
 
 #initialise frequencies  
 fnyq = 0.5/dtm
 flo = 1.0/(tghi - tglo)
 fhi = fnyq 
 if (fbreak == -1):
  w0  = min(10*flo,fhi/2)*twopi
 else:
  w0 = twopi*fbreak
  
 if (custom_freqs == []): 
  f = np.arange(flo,fhi+flo,flo)
 else:
  f = custom_freqs
 
 w =f*twopi
 nw = np.shape(w)[0]
 np2 = 2*nw
 nptot = np2 + order + 1
 #define power spectrum prior
 w2 = (w/w0)**(-2)
 w2_integrate = w0**2 * (1/w[0] - 1/w[-1])
 #w2 = 1./(1+(w/w0)**2)
 #w2_integrate = w0*(1./np.tan(w[-1]/w0) - 1./np.tan(w[0]/w0))

 
 #bplin=[fbreak,2,2]
 #w2 = (1. + (w/(twopi*bplin[0]))**(bplin[1]-bplin[2])) / (w/(twopi*bplin[0]))**bplin[1]
 #
 #w_hi = w[-1]
 #w_lo = w[0]
 #w2_integrate_hi = w_hi*( (w_hi/(twopi*bplin[0]))**(-bplin[1])/(1.-bplin[1]) + (w_hi/(twopi*bplin[0]))**(-bplin[2])/(1.-bplin[2]) )
 #w2_integrate_lo = w_lo*( (w_lo/(twopi*bplin[0]))**(-bplin[1])/(1.-bplin[1]) + (w_lo/(twopi*bplin[0]))**(-bplin[2])/(1.-bplin[2]) )
 #w2_integrate = w2_integrate_hi - w2_integrate_lo
 
 
 
 
 
 
 
 #save the sine and cosine value in  2d array to speed computation
 cwt = np.ones((ny,nw))
 swt = np.ones((ny,nw))
 polyt = np.ones((ny,order + 1))

 tp = np.mean(t)
 for iw in range(order + 1):
  polyt[:,iw] = (t - tp)**iw

 for iw in range(nw):
  cwt[:,iw] = np.cos(w[iw]*t)
  swt[:,iw] = np.sin(w[iw]*t)
 
 
 #perform hessian fit
 cvec = np.ones(nptot)
 hes = np.ones((nptot,nptot))
 s2 = s*s
 cwtT = cwt.T
 swtT = swt.T
 polytT = polyt.T
 
 hnow_cc = np.tensordot(cwtT/s2,cwt,axes=1)
 hnow_sc = np.tensordot(swtT/s2,cwt,axes=1)
 hnow_cs = np.tensordot(cwtT/s2,swt,axes=1)
 hnow_ss = np.tensordot(swtT/s2,swt,axes=1)
 hnow_cp = np.tensordot(cwtT/s2,polyt,axes=1)
 hnow_pc = np.tensordot(polytT/s2,cwt, axes=1)
 hnow_sp = np.tensordot(swtT/s2,polyt,axes=1)
 hnow_ps = np.tensordot(polytT/s2,swt, axes=1)
 hnow_pp = np.tensordot(polytT/s2,polyt, axes=1)
 
 
 idk = np.arange(0,np2,2)
 idk1 = np.arange(1,np2,2)
 hes[:np2:2,:np2:2] = hnow_cc[:,:].T
 hes[:np2:2,1:np2:2] = hnow_sc[:,:].T
 hes[1:np2:2,:np2:2] = hnow_cs[:,:].T
 hes[1:np2:2,1:np2:2] = hnow_ss[:,:].T
 hes[np2:,:np2:2] = hnow_cp.T
 hes[:np2:2,np2:] = hnow_pc.T
 hes[np2:,1:np2:2] = hnow_sp.T
 hes[1:np2:2,np2:] = hnow_ps.T
 hes[np2:,np2:] = hnow_pp.T

 ##add on prior unless fitting custom frequencies 
 if (custom_freqs == []):
  x = np.sqrt(w2_integrate)/(2*rms*w2)
 else:
  x = 0
  
 hes[idk,idk] = hes[idk,idk] + x
 hes[idk1,idk1] = hes[idk1,idk1] + x
 ys2 = y/s2
 cvec[idk] = np.dot(cwtT,ys2)
 cvec[idk1] = np.dot(swtT,ys2)
 cvec[np2:] = np.dot(polytT,ys2)
 
 
 #t1 = time.time()
 #iwp = 0
 #for ikp in range(0,np2,2):
 # wp = w[iwp]
 # wp2 = wp*wp
 # cvec[ikp]   = np.sum(y*cwt[:,iwp]/s2)
 # cvec[ikp+1] = np.sum(y*swt[:,iwp]/s2)
 # iw = 0
 # for ik in range(0,np2,2):
 #  #cosine column
 #  hes[ikp,ik]   = hnow_cc[iw,iwp]
 #  hes[ikp,ik+1] = hnow_sc[iw,iwp]
 #  #sine column
 #  hes[ikp+1,ik]   = hnow_cs[iw,iwp]
 #  hes[ikp+1,ik+1] = hnow_ss[iw,iwp]
 #  #priors
 #  if (ik == ikp):
 #   hes[ikp,ik] = hnow_cc[iw,iwp] +  np.sqrt(w2_integrate)/(2*rms*w2[iwp]) 
 #   hes[ikp+1,ik+1] = hnow_ss[iw,iwp] + np.sqrt(w2_integrate)/(2*rms*w2[iwp]) 
 #  iw = iw + 1  
 # iwp = iwp + 1
 #t2 = time.time()
 #print('time2',t2-t1)
 
 #print('hes')
 #for i in range(5):
 # print(hes[i,:5])
 # 
 # 
 #print('hes2')
 #for i in range(5):
 # print(hes2[i,:5])
 #input()
 
 #invert the matrix to get the covariance matrix
 cov = np.linalg.inv(hes)
 

 
 #find the parameters
 parm  = cov.dot(cvec)
 skout = parm[1:np2:2]
 ckout = parm[0:np2:2]
 poly_coef = parm[np2:]


 ymodout = np.dot(swt,skout) + np.dot(cwt,ckout) + np.dot(polyt,poly_coef)
 var = np.var(ymodout - y)
 
 if si is None:
  nparm = np.shape(parm)[0]
  ndof = max(1, ny - nparm)
  expansion = np.sum((ymodout - y) ** 2) / ndof
  cov = cov * expansion

 
 #monte carlo sample the fit to estimate uncertaintie
 cwhires = np.ones((ntgrid,nw))
 swhires = np.ones((ntgrid,nw))
 polyt_hires = np.ones((ntgrid,order+1))
 for iw in range(nw):
  cwhires[:,iw] = np.cos(w[iw]*tgrid)
  swhires[:,iw] = np.sin(w[iw]*tgrid)
 for iw in range(order+1):
  polyt_hires[:,iw] = (tgrid - tp)**iw

 parmnow = np.random.multivariate_normal(parm,cov,size=nits)
 ygridsave = np.zeros((ntgrid,nits))

 if nits > 1:
  for i in range(nits):
   parmnew = parmnow[i,:]
   cknew = parmnew[0:np2:2]
   sknew = parmnew[1:np2:2]
   poly_coef_new = parmnew[np2:]
   ygridsave[:,i] = np.dot(swhires,sknew) + np.dot(cwhires,cknew) + np.dot(polyt_hires,poly_coef_new)
  ygridsave = ygridsave*ystd + ymean
  ygridlo,ygrid,ygridhi = np.percentile(ygridsave,[15.865,50,84.135],axis=1)
 else:
  ygrid = (np.dot(swhires,parm[1:np2:2]) + np.dot(cwhires,parm[0:np2:2]) + np.dot(polyt_hires,parm[np2:])) * ystd + ymean
  sd = np.sqrt(var)
  ygridlo,ygridhi = ygrid-sd, ygrid + sd

 ygridop = np.array([ygridlo,ygrid,ygridhi]).T
 
 
 #fit stat
 ymod_itp = np.interp(ti,tgrid,ygrid)
 rcoef = np.corrcoef(ymod_itp,yi)[0,1]
 

 return(tgrid,ygridop,f,ckout*ystd,skout*ystd,poly_coef*ystd,rcoef)
 
 
 
 




#
##test
#import matplotlib.pylab as plt
#t = np.arange(100)
#x = 0.1*t + 0.1*np.sin(2*np.pi/20*t)
#
#tgrid = np.arange(150)
#tgrid,ygridop,f,ckout,skout,polyout,rcoef = rw(t,x,si=0,tgi = tgrid,fbreak=-1,custom_freqs=np.array([1./20]),nits = 1,order = 2)
#
#
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(tgrid,ygridop[:,1],label='model')
#ax1.plot(t,x,label='data')
#plt.legend()
#plt.show()