import numpy as np
import time
dat = np.array([[3.,2.,4.],[2.,-3.,1.],[1.,1.,2.]])

tstart = time.time()
dinv = np.linalg.inv(dat)
tend = time.time()
print('inversion time PYTHON=',tend - tstart)


import os
os.system('./a.out')


#now test f2py ability to call fortran from python
os.system('f2py -c -m my_lib greedy_select_f90.f90')
import greedy_select_f90 as gs

f90inv = gs.inverse(dat)