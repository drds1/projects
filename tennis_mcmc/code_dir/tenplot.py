import matplotlib.pylab as plt
import numpy as np
import scipy.interpolate
import os
import matplotlib

font = {'family' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

scorelo = 0
scorehi = 7
point_win = 0.55
prob_point_set_study = 1





playernames=['Player 1','Player 2']


os.system('gfortran calc_tennis.f90 -o calc_tennis.exe')
#with open('calc_tennis.par') as f:
# mylist = f.read().splitlines()
#mylist[-1] = np.str(point_win)+'\n'
#f.close()
f = open('calc_tennis.par','w')
f.write('4,6\n')
f.write(np.str(point_win)+'\n')
f.close()
#f.write('\n'.join(mylist)+'\n')
os.system('./calc_tennis.exe')



dat = np.loadtxt('mc_tennis_posterior.txt')


ndat = np.shape(dat[:,0])[0]
N = int(np.sqrt(ndat))
z =dat[:,2].reshape(N, N)


fig=plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist2d(dat[:,0],dat[:,1])
ax1.set_xlabel(playernames[0]+' score')
ax1.set_ylabel(playernames[1]+' score')
ax1.plot([scorelo,scorehi],[scorelo,scorehi],ls='--',color='blue',label=None)
ax1.text(0.3,0.3,playernames[0]+' win',ha='center',rotation=45,transform=ax1.transAxes,color='b',fontsize=14)
ax1.text(0.3,0.5,playernames[1]+' win',ha='center',rotation=45,transform=ax1.transAxes,color='b',fontsize=14)

cblab = 'result probability'
pic = ax1.imshow(np.flipud(z),interpolation='gaussian',extent=[scorelo,scorehi]*2)
pic.set_cmap('hot')
fig.colorbar(pic,label=cblab)
plt.title(playernames[0]+' point-win probability = '+np.str(point_win))

#plt.colorbar()


plt.savefig('tenplot.pdf')






#now compute the probability of winning a set vs the probability of winning a rally
#to see how a slight increase in skill compounds through a game
if (prob_point_set_study == 1):

 pw_vals = np.arange(0.1,0.9,0.01)
 
 nvals = np.shape(pw_vals)[0]
 pw_win = np.zeros(nvals)
 
 for i in range(nvals):
  pwnow = pw_vals[i]
  #with open('calc_tennis.par') as f:
  #   mylist = f.read()
  #mylist[-1] = np.str(pwnow)+'\n'
  #f.close()
  #f = open('calc_tennis.par')
  #f.write(''.join(mylist))
  
  f = open('calc_tennis.par','w')
  f.write('4,6\n')
  f.write(np.str(pwnow)+'\n')
  f.close()
  os.system('./calc_tennis.exe')
 
  dat = np.loadtxt('mc_tennis_posterior.txt')
  ndat = np.shape(dat[:,0])[0]
  N = int(np.sqrt(ndat))
  z =dat[:,2].reshape(N, N)
  with open('calc_tennis.par') as f:
      mylist = f.read().splitlines() 
  f.close()
  winpoint_prob = np.float(mylist[-1])
  
  x = np.arange(scorehi)
  y = np.arange(scorehi)
  prob_itp = scipy.interpolate.interp2d(x, y, z,kind='cubic')
  #player 1 win
  sum1 = 0.0
  sum2 = 0.0
  i1res = np.linspace(0,scorehi,100)
  i2res = np.linspace(0,scorehi,100)
  for i1 in i1res:
   for i2 in i2res:
    pnow = prob_itp(i1,i2)[0]
    if (i1 > i2):
     sum1 = sum1 + pnow
    else:
     sum2 = sum2 + pnow
  ptot = sum1 + sum2
  p1win = sum1/ptot
  p2win = sum2/ptot
  
  pw_win[i] = p1win
  print 'point win prob ',pwnow,' set win prob ',p1win
 
 #cannot have probabilities greater than 1. Some slight numerical bug. Quick fix
 #no huge effect but fix correctly later
 pw_win = np.minimum(pw_win,np.ones(nvals))
 pw_lose = 1.*pw_win 
 pw_win = 1.-pw_win
 
 f = open('point_set_win_prob.txt','w')
 for i in range(nvals):
  f.write(np.str(pw_vals[i])+','+np.str(pw_win[i])+','+np.str(pw_lose[i])+'\n')
 f.close()
 
 
 #now make python plot
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 
 ax1.plot(pw_vals,pw_win,label='win probability')
 ax1.plot(pw_vals,pw_lose,label='lose probability')
 ax1.set_xlabel('point win probability')
 ax1.set_ylabel('set win probability')
 ax1.plot([0,1],[0,0],label=None,ls='--',color='k')
 ax1.plot([0,1],[1,1],label=None,ls='--',color='k')
 ax1.legend(fontsize='small')
 plt.savefig('fig_prob_point_set.pdf')
 


 