import numpy as np
import matplotlib.pylab as plt
import math



def poisson(actual, mean):
    return math.pow(mean, actual) * math.exp(-mean) / math.factorial(actual)


pn = np.vectorize(poisson)

k = [1,2,4,8]

x = np.arange(20)

col = ['b','r','g','c']

nk = len(k)
fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in range(nk):
 
 
 know = k[i]
 y  = pn(x,know)
 ysum = np.sum(y)
 
 if (i == 0):
  ax1.plot(x,y/ysum,label=r'$\langle \mu \rangle$ = '+np.str(know)+' (mean number of corners per game)' ,marker='o',color=col[i])
 else:
  ax1.plot(x,y/ysum,label=r'$\langle \mu \rangle$ = '+np.str(know),marker='o',color=col[i])
 
 ax1.plot([know]*2,[0,pn(know,know)],label=None,color=col[i])

ax1.set_xticks(x[::2])
ax1.set_ylabel('probability density')
ax1.set_xlabel('number of events')
plt.legend() 
plt.savefig('fig_poison.pdf')