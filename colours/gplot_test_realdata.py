import numpy as np


file = './Mrk509/output_20180723_001/cream_furparms.dat'
with open(file) as f:
 content = f.readlines()
 content = [x.strip() for x in content] 
 f.close()
x = [np.float(cnow) for cnow in content[0].split()]
p0new,dwnew,w0new = x
dfur = np.loadtxt(file,skiprows=1)
fnow,sfur,cfur,sd_sfur,sd_cfur = dfur.T
nf = np.shape(fnow)[0]
nf2 = 2*nf
pnew = np.zeros(nf2)
sdnew = np.zeros(nf2)
idx = 0
for i in range(0,nf2,2):
 pnew[i] = cfur[idx]
 pnew[i+1] = sfur[idx]
 sdnew[i]= sd_cfur[idx]
 sdnew[i+1] = sd_sfur[idx]
 idx = idx + 1
 
mgp.gp(pnew,sdnew,fnow,p0=p0new,dw=dwnew,w0=w0new,plot_tit='')
