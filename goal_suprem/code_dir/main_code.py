import numpy as np
#import code_pythonsubs as cpc
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import math





#DATA preparation
features = pd.read_csv('samples.csv')
Es = features.values[:,1]
Et = features.values[:,2]
ht_h = features.values[:,3]
ht_a = features.values[:,4]
ft_h = features.values[:,5]
ft_a = features.values[:,6]


#define poisson distribution
def poisson(actual, mean):
    return math.pow(mean, actual) * math.exp(-mean) / math.factorial(actual)
pv = np.vectorize(poisson)







#1) How to convert the expected goal supremacy and total goals into an expected full time 
#goals for the home and away teams

#Given these quantities...
#Expected goal supremacy E(s) = E(H-A)
#Expected total number of goals E(t) = E(H+A)

#Therefore, expected Home goals
#E(H) = (E(t) + E(s))/2

#... and expected away goals
#E(A) = (E(t) - E(s))/2

Eh = 0.5*(Et + Es)
Ea = 0.5*(Et - Es)



#plot these quantities as histogram
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(Eh,bins=50,label='Home Expected Goals',histtype='step')
ax1.hist(Ea,bins=50,label='Away Expected Goals',histtype='step')
#overlay actual full time goals
#ax1.hist(ft_h,bins=50,label='Home Actual Goals',histtype='step')
#ax1.hist(ft_a,bins=50,label='Away Actual Goals',histtype='step')
ax1.set_xlabel('Expected goals')
ax1.set_ylabel('Number')
plt.legend()
plt.savefig('total_goals_hist.pdf')


#2) Why is Poisson distribution good for this type of game (see attached writeup)













#3) General game double poisson distribution for home and away team
# want probability of particular scores (consider scores up to 9)
interested_score = [0,0]
smax = 9
Eh_now = 1.6
Ea_now = 0.9

#generate an scoresXscores array of possible score combinations and fit these with
#poisson distribution
scores = np.arange(smax+1)
hnow,anow = np.array(np.meshgrid(scores,scores))
P_h = pv(hnow,Eh_now)
P_a = pv(anow,Ea_now)
P_tot = P_h*P_a
P_sum = np.sum(P_tot)

idx,idy = np.where((hnow == interested_score[0]) & (anow == interested_score[1]))
idx=idx[0]
idy=idy[0]
print 'probability of (H,A) score:',hnow[idx,idy],anow[idx,idy],'is ',P_tot[idx,idy]




#4) now compute probability of Home Victory
xh_win,yh_win = np.array(np.where(hnow > anow))
p_h_win = np.sum(P_tot[xh_win,yh_win])
print 'Probability of home victory:',p_h_win

#for i1 in range(smax+1):
# for i2 in range(smax+1):
#  print hnow[i1,i2],anow[i1,i2],P_tot[i1,i2]






#plot example posterior probability distribution
P_h_f = anow.flatten()
P_a_f = hnow.flatten()
P_tot_f = P_tot.flatten()

plt.clf()
N = int(len(P_tot_f)**.5)
z = P_tot_f.reshape(N, N)
#extent=(np.amin(P_h_f), np.amax(P_h_f), np.amin(P_a_f), np.amax(P_a_f))
plt.imshow(z,extent=(0,10,0,10) ,aspect=1,origin='lower')
cbar = plt.colorbar()
plt.xlabel(r'$i$ Home Goals')
plt.ylabel(r'$j$ Away Goals')
cbar.set_label(r'2D probability distribution $P_{ij}$')  
plt.savefig('fig_eg_posterior.pdf')













#5) How to calculate expected goals at half time from scores of each team
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(ft_h,label='Ft Home Goals',histtype='step')
ax1.hist(ft_a,label='Ft Away Goals',histtype='step')
ax1.hist(ht_h,label='Ht Home Goals',histtype='step')
ax1.hist(ht_a,label='Ht Away Goals',histtype='step')
ax1.set_xlabel('Expected goals')
ax1.set_ylabel('Number')
plt.legend()
plt.savefig('total_goals_hist.pdf')
plt.savefig('fig_halftime_fulltime_actuall.pdf')
##   
##   

fig = plt.figure()
ax1 = fig.add_subplot(111)
x = np.arange(10)
ax1.hist(ft_h-ht_h,bins=x,label='Second half home',histtype='step')
ax1.hist(ft_a-ht_a,bins=x,label='Second half away',histtype='step')
ax1.hist(ht_h,bins=x,label='First half home',histtype='step')
ax1.hist(ht_a,bins=x,label='First half away',histtype='step')
plt.legend()
plt.savefig('fig_1st_2nd_comp.pdf')

col = ['r','b']
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = np.arange(10)

xh2 = np.append(ft_h-ht_h,ft_a-ht_a)
ax1.hist(xh2,bins=x,label='Second half',color=col[1],histtype='step')
xh1 = np.append(ht_h,ht_a)
ax1.hist(xh1,bins=x,label='First half',color=col[0],histtype='step')
ylim = list(ax1.get_ylim())

xh2mean = np.mean(xh2)
xh1mean = np.mean(xh1)

#This is the infamous X value where E(2) = X E(1) therefore E(1) = T/(1+X)
X = xh2mean/xh1mean
ax1.plot([xh1mean]*2,ylim,color=col[0],label=None)
ax1.plot([xh2mean]*2,ylim,color=col[1],label=None)
ax1.set_xlabel('Home + Away goals scored')
ax1.text(0.98,0.8,'X = E(2)/E(1) = '+np.str(np.round(X,2)),ha='right',transform=ax1.transAxes)
plt.legend()
plt.savefig('fig_1st_2nd_comp_tot.pdf')









#6) See the writeup.pdf file but E(1) = T/(1+X) relating expected goals at 
#hlaf time to expected goals T at full time.




#7) Devise a joint probability distribution for the goals in the first half separately 
#using definition of X = E(2)/E(1) as the ratio of the expected goals in the first
#and second halves respectively 
#What is prob of home team winning 1-0 at half time then 2-1 at full time?
#use example home and away full time expectation values form Q3
scoreht = [1,0]
scoreft = [2,1]


#use the same hnow and anow array designed for question 3 and the expectation values from question 1
Eh1_now = Eh_now/(1.+X)
Ea1_now = Ea_now/(1.+X)
P_h1 = pv(hnow,Eh1_now)
P_a1 = pv(anow,Ea1_now)
P_tot_1 = P_h1*P_a1
P_sum_1 = np.sum(P_tot_1)

idx1,idy1 = np.where((hnow == scoreht[0]) & (anow == scoreht[1]))
idx1=idx1[0]
idy1=idy1[0]
print 'probability of half time (H,A) score:',hnow[idx1,idy1],anow[idx1,idy1],'is ',P_tot_1[idx,idy]



#now compute again for the 2nd half now T = E(1) + E(2) = E(2)(1 + 1/X)
Eh2_now = Eh_now/(1.+1./X)
Ea2_now = Ea_now/(1.+1./X)
P_h2 = pv(hnow,Eh2_now)
P_a2 = pv(anow,Ea2_now)
P_tot_2 = P_h2*P_a2
P_sum_2 = np.sum(P_tot_2)

idx2,idy2 = np.where((hnow == scoreft[0]-scoreht[0]) & (anow == scoreft[1] - scoreht[1]))
idx2=idx2[0]
idy2=idy2[0]
print 'probability of scoring another ',hnow[idx2,idy2],anow[idx2,idy2],\
'goals in 2nd half is',P_tot_2[idx2,idy2]
print 'probability of full time (H,A) score:',hnow[idx1,idy1]+hnow[idx2,idy2],\
anow[idx1,idy1]+anow[idx2,idy2],'is ',P_tot_1[idx,idy]*P_tot_2[idx2,idy2]









#8) Now calculate summed probability of a draw in first half and an away team win in second half
#draw in first half
id1draw_x,id1draw_y = np.where(anow == hnow)
p1draw = np.sum(P_tot_1[id1draw_x,id1draw_y])

#calculate the other outcomes out of curiosity
id1win_x,id1win_y = np.where(hnow>anow)
p1win = np.sum(P_tot_1[id1win_x,id1win_y])
id1lo_x,id1lo_y = np.where(hnow<anow)
p1lo = np.sum(P_tot_1[id1lo_x,id1lo_y])

#home lose in second half (normally need to remember the scores in first half but since
#we know to look for a draw, the scores at half time will be level)
id2lo_x,id2lo_y = np.where(hnow<anow)
p2lo = np.sum(P_tot_2[id2lo_x,id2lo_y])

print 'probability of draw in first half',p1draw,'. Probability of away team win in 2nd half',\
p2lo,'. Joint probability da',p1draw*p2lo














#9) NOw load the test data and compute the double probabilities

#DATA preparation
features = pd.read_csv('test.csv')
test_Es = features.values[:,1]
test_Et = features.values[:,2]
ntest = np.shape(test_Es)[0]

#compute all the required parameters for the joint posterior probability distributions
#after the first half and after full time
test_Eh = 0.5*(test_Et + test_Es)
test_Ea = 0.5*(test_Et - test_Es)

test_Eh1_now = test_Eh/(1.+X)
test_Ea1_now = test_Ea/(1.+X)

test_Eh2_now = test_Eh/(1.+(1./X))
test_Ea2_now = test_Ea/(1.+(1./X))



#now have 4D combinations of scores to consider
h1,a1,h2,a2 = np.array(np.meshgrid(scores,scores,scores,scores))







#initiate loop for each game
data = []
for it in range(ntest):
 tEh1 = test_Eh1_now[it]
 tEa1 = test_Ea1_now[it]
 tEh2 = test_Eh2_now[it]
 tEa2 = test_Ea2_now[it]
 
 psave = np.zeros((0,4))
  
 for i1 in range(smax+1):
  for i2 in range(smax+1):
   for i3 in range(smax+1):
    for i4 in range(smax+1):   
     psave = np.vstack((psave,np.array([i1,i2,i3,i4])))
 nps = np.shape(psave)[0]
 prob = []
 for ip in range(nps):
  tP_h1 = poisson(psave[ip,0],tEh1)
  tP_a1 = poisson(psave[ip,1],tEa1)
  tP_h2 = poisson(psave[ip,2],tEh2)
  tP_a2 = poisson(psave[ip,3],tEa2)
  prob.append(tP_h1*tP_a1*tP_h2*tP_a2)
  
 prob = np.array(prob)
 psavesum = np.sum(prob)
 prob = np.array(prob)/psavesum
 
 #hh,hd,ha,dh,dd,da,ah,ad,aa
 
 #P(hh)
 idx_hh = np.where((psave[:,0] > psave[:,1]) & ((psave[:,0] + psave[:,2]) > (psave[:,1] + psave[:,3])))
 sum_hh = np.sum(prob[idx_hh])
 
 #P(hd)
 idx_hd = np.where((psave[:,0] > psave[:,1]) & ((psave[:,0] + psave[:,2]) == (psave[:,1] + psave[:,3])))
 sum_hd = np.sum(prob[idx_hd])

 #P(ha)
 idx_ha = np.where((psave[:,0] > psave[:,1]) & ((psave[:,0] + psave[:,2]) < (psave[:,1] + psave[:,3])))
 sum_ha = np.sum(prob[idx_ha])
 
 #P(dh)
 idx_dh = np.where((psave[:,0] == psave[:,1]) & ((psave[:,0] + psave[:,2]) > (psave[:,1] + psave[:,3])))
 sum_dh = np.sum(prob[idx_dh])
 
 #P(dd)
 idx_dd = np.where((psave[:,0] == psave[:,1]) & ((psave[:,0] + psave[:,2]) == (psave[:,1] + psave[:,3])))
 sum_dd = np.sum(prob[idx_dd])
 
 #P(da)
 idx_da = np.where((psave[:,0] == psave[:,1]) & ((psave[:,0] + psave[:,2]) < (psave[:,1] + psave[:,3])))
 sum_da = np.sum(prob[idx_da])
 
 #P(ah)
 idx_ah = np.where((psave[:,0] < psave[:,1]) & ((psave[:,0] + psave[:,2]) > (psave[:,1] + psave[:,3])))
 sum_ah = np.sum(prob[idx_ah])
 
 #P(ad)
 idx_ad = np.where((psave[:,0] < psave[:,1]) & ((psave[:,0] + psave[:,2]) == (psave[:,1] + psave[:,3])))
 sum_ad = np.sum(prob[idx_ad]) 

 #P(aa)
 idx_aa = np.where((psave[:,0] < psave[:,1]) & ((psave[:,0] + psave[:,2]) < (psave[:,1] + psave[:,3])))
 sum_aa = np.sum(prob[idx_aa]) 
 #P(hd)
 
 
 print features.values[it,0],sum_hh,sum_hd,sum_ha,sum_dh,sum_dd,sum_da,sum_ah,sum_ad,sum_aa
 data.append([features.values[it,0],sum_hh,sum_hd,sum_ha,sum_dh,sum_dd,sum_da,sum_ah,sum_ad,sum_aa])
 #print psavesum,np.sum(prob),sum_hh+sum_hd+sum_ha+sum_dh+sum_dd+sum_da+sum_ah+sum_ad+sum_aa
 print ''
 
 
 
#output the results as a pandas data frame to predictions.csv
print 'saving results to csv file predictions.csv'
df = pd.DataFrame(np.array(data), columns=['ID','hh','hd','ha','dh','dd','da','ah','ad','aa'])
df.to_csv('predictions.csv')
 
 
 
 
 
 
 
#miscellaneous scripts 
 
 #idx_d1 = np.where(psave[:,0] == psave[:,1])[0]
 #sum_d1 = np.sum(psave[idx_d1,-1])
 #
 ##compute prob of homewin after 1st half
 #idx_h1 = np.where(psave[:,0] > psave[:,1])[0]
 #sum_h1 = np.sum(psave[idx_h1,-1]) 
#
 ##compute prob of away win after 1st half
 #idx_a1 = np.where(psave[:,0] < psave[:,1])[0]
 #sum_a1 = np.sum(psave[idx_a1,-1]) 
 
 
 #compute prob 
 
 ##compute the possible scores for 1st half
 #tP_h1 = pv(hnow,tEh1)
 #tP_a1 = pv(anow,tEa1)
 #tP_tot1 = tP_h1*tP_a1
 ##home win possibility in first half
 #idh1 = np.where(hnow>anow)
 #sum_h1 = np.sum(tP_tot1[idh1[0],idh1[1]])
 #
 #
 ##draw probability in first half
 #idd1 = np.where(hnow==anow)
 #sum_d1 = np.sum(tP_tot1[idd1[0],idd1[1]])
 #
 ##away win possibility in first half
 #ida1 = np.where(hnow<anow)
 #sum_a1 = np.sum(tP_tot1[ida1[0],ida1[1]])


 #print 'team id',features.values[it,0]
 #print '1st half P(w)',sum_h1,'P(d)',sum_d1,'P(a)',sum_a1



 ##second half
 ##compute the possible scores after full time from the original expectation
 ##values from Question 1
 #tP_h_ft = pv(hnow,tEh_ft)
 #tP_a_ft = pv(anow,tEa_ft)
 #tP_tot_ft = tP_h_ft*tP_a_ft
 ##home win possibility in first half
 #idh1 = np.where(hnow>anow)
 #sum_h1 = np.sum(tP_tot1[idh1[0],idh1[1]])
 ##draw probability in first half
 #idd1 = np.where(hnow==anow)
 #sum_d1 = np.sum(tP_tot1[idd1[0],idd1[1]])
 ##away win possibility in first half
 #ida1 = np.where(hnow<anow)
 #sum_a1 = np.sum(tP_tot1[ida1[0],ida1[1]])

 
 
 #each score combination from the first half will have different
 #score combinations in the second half that constitute a win
 


 

 #print ''


fig = plt.figure()
ax1 = fig.add_subplot(111)
x = np.arange(10)
xh2 = (ft_h-ht_h)-(ft_a-ht_a)
ax1.hist(xh2,bins=x,label='Second half',color=col[1],histtype='step')
xh1 = ht_h-ht_a
ax1.hist(xh1,bins=x,label='First half',color=col[0],histtype='step')
ylim = list(ax1.get_ylim())
xh2mean = np.mean(xh2)
xh1mean = np.mean(xh1)
ax1.plot([xh1mean]*2,ylim,color=col[0],label=None)
ax1.plot([xh2mean]*2,ylim,color=col[1],label=None)
ax1.set_xlabel('Supremecy goals scored')
ax1.text(0.98,0.8,'X = E(2)/E(1) = '+np.str(np.round(xh2mean/xh1mean,2)),ha='right',transform=ax1.transAxes)
plt.legend()
plt.savefig('fig_1st_2nd_comp_sup.pdf')





fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(ft_h,Eh,label='home',marker='o',ls='')
ax1.plot(ft_a,Ea,label='away',marker='o',ls='')
xl = list(ax1.get_xlim())
yl = list(ax1.get_ylim())
ax1.set_xlabel('expected')
ax1.set_ylabel('actuall')
ax1.plot(xl,xl,ls='--',color='k')
plt.legend()
plt.savefig('expected_vs_actual_ft.pdf') 


