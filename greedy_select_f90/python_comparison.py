import numpy as np
import time
from sklearn.linear_model import LinearRegression
import os

'''
test inverse matrix
dat = np.array([[3.,2.,4.],[2.,-3.,1.],[1.,1.,2.]])

tstart = time.time()
dinv = np.linalg.inv(dat)
tend = time.time()
print('inversion time PYTHON=',tend - tstart)
os.system('./a.out')
'''



'''
test f2py

#now test f2py ability to call fortran from python
os.system('f2py -c -m my_lib greedy_select_f90.f90')
import greedy_select_f90 as gs
f90inv = gs.inverse(dat)
'''


'''
test glm
'''
n = 2000
k = 100

ptrue = [20.0,13.0]

t = np.arange(1,n+1)

#make fake datsa identical to fortran
y = 3.2*np.sin(2*3.1415926535/ptrue[0] * t) + 8.0*np.sin(2*3.1415926535/ptrue[1] * t)
X = np.zeros((n,k))
for ik in range(k):
    X[:,ik] = np.random.randn(n)

for ik in range(len(ptrue)):
    X[:,ik] = np.sin(2*3.1415926535/ptrue[ik] * t)


f = LinearRegression()
tf1 = time.time()
print(np.shape(X),np.shape(y),'datashape')
f.fit(X,y)
tf2 = time.time()

tp1 = time.time()
pred = f.predict(X)
tp2 = time.time()

print('training time',tf2-tf1)
print('predicting time',tp2-tp1)
print('ratio',(tf2-tf1)/(tp2-tp1))
print('cisq',np.sum((pred - y)**2))
print('NOW F90')
os.system('gfortran greedy_select_f90.f90 f90random.f90')
os.system('./a.out')

