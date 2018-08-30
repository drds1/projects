import numpy as np
import myfitrw_current as mfrw




iskip = 10
dat = np.loadtxt('labels_info.csv',delimiter=',')

t = dat[::iskip,0]
y = dat[::iskip,1]

ny = y.size
sig = np.ones(ny)


a = mfrw.fitrw([t],[y],[sig],floin=-1,fhiin=-1,ploton=1,dtresin=-1,nits = 1)



