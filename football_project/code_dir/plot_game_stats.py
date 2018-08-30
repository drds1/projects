import csv, math, ast 
import numpy as np
import matplotlib.pylab as plt
import json



s = open('dict_dump.txt', 'r').read()
dict = ast.literal_eval(s)



	
#plot number of home and away games for each team
teamid = np.array([np.float(dn) for dn in dict.keys()])
n_home = np.array(	[dict[dn]['home_games'] for dn in dict.keys()] )
n_away = np.array(	[dict[dn]['away_games'] for dn in dict.keys()] )
#league_id_train = np.array(	[dict[dn]['league_id'] for dn in dict.keys()] )
#league_id_train_u = np.unique(league_id_train)
alpha_a_train = np.array(	[dict[dn]['alpha_a'] for dn in dict.keys()] )
alpha_h_train = np.array(	[dict[dn]['alpha_h'] for dn in dict.keys()] )
beta_a_train = np.array(	[dict[dn]['beta_a'] for dn in dict.keys()] )
beta_h_train = np.array(	[dict[dn]['beta_h'] for dn in dict.keys()] )


#!!!!!!!!!!plot the alpha and beta values (averages for each league id from training set)


league_id_train = []
#team_id_train_home = []
#team_id_train_away = []
corner_home = []
corner_away = []
nlt = np.shape(teamid)[0]
csvFile = 'train.csv'
csvRead = csv.reader(open(csvFile))
next(csvRead)
ave_home,ave_away = np.loadtxt('avg_home_away.txt')
for game in csvRead:
 league_id_train.append(np.float(game[1]))
 #team_id_train_home.append(np.float(game[3]))
 #team_id_train_away.append(np.float(game[4]))
 corner_home.append(np.float(game[7]))
 corner_away.append(np.float(game[8]))
league_id_train = np.array(league_id_train)
league_id_train_u = np.unique(league_id_train)
corner_home = np.array(corner_home)
corner_away = np.array(corner_away)

nlu = np.shape(league_id_train_u)[0]
nlt = np.shape(corner_home)[0]


a_home_league = np.zeros(nlu)
b_home_league = np.zeros(nlu)
n_league_count = np.zeros(nlu)
a_away_league = np.zeros(nlu)
b_away_league = np.zeros(nlu)

a2_home_league = np.zeros(nlu)
b2_home_league = np.zeros(nlu)
a2_away_league = np.zeros(nlu)
b2_away_league = np.zeros(nlu)

for il in range(nlu):
 lunow = league_id_train_u[il]
 for it in range(nlt):
  
  lttrain = league_id_train[it]
  ch_now = corner_home[it]
  ca_now = corner_away[it]
  if (lttrain == lunow):
   n_league_count[il] = n_league_count[il] + 1
   a_home_league[il] = a_home_league[il] + ch_now
   b_home_league[il] = b_home_league[il] + ca_now
   a_away_league[il] = a_away_league[il] + ca_now
   b_away_league[il] = b_away_league[il] + ch_now
   a2_home_league[il] = a2_home_league[il] + ch_now**2
   b2_home_league[il] = b2_home_league[il] + ca_now**2
   a2_away_league[il] = a2_away_league[il] + ca_now**2
   b2_away_league[il] = b2_away_league[il] + ch_now**2
   
   
a_home_league = np.array(a_home_league) 
b_home_league = np.array(b_home_league)   
a_away_league = np.array(a_away_league) 
b_away_league = np.array(b_away_league) 
ave_a_home = a_home_league/n_league_count/ave_home  
ave_b_home = b_home_league/n_league_count/ave_away   
ave_a_away = a_away_league/n_league_count/ave_away    
ave_b_away = b_away_league/n_league_count/ave_home 

sd_a_home = (a2_home_league/n_league_count - (a_home_league/n_league_count)**2)**0.5/ave_home/2
sd_b_home = (b2_home_league/n_league_count - (b_home_league/n_league_count)**2)**0.5/ave_away/2
sd_a_away = (a2_away_league/n_league_count - (a_away_league/n_league_count)**2)**0.5/ave_away/2
sd_b_away = (b2_away_league/n_league_count - (b_away_league/n_league_count)**2)**0.5/ave_home/2


#plot the average alpha and beta distribution across leagues
xave = np.arange(nlu)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.errorbar(xave,ave_a_home,sd_a_home,marker='o',label=r'$\alpha$ home')
ax1.errorbar(xave,ave_a_away,sd_a_away,marker='o',label=r'$\alpha$ away')
ax1.errorbar(xave,ave_b_home,sd_b_home,marker='o',label=r'$\beta$ home')
ax1.errorbar(xave,ave_b_away,sd_b_away,marker='o',label=r'$\beta$ away')
ax1.set_xlabel('league ID')
ax1.set_ylabel(r'$\alpha$ or $\beta$ value')

xtl = [np.str(np.int(xn)) for xn in league_id_train_u] 
#ax1.set_xticks(xave,rotation='vertical')
#ax1.set_xticklabels(xtl)
plt.xticks(xave, xtl, rotation='vertical')
plt.legend()
plt.savefig('fig_alpha_beta_league_ave.pdf')


f = open('alpha_beta_league_stats.txt','w')
for i in range(nlu):
 f.write(np.str(league_id_train_u[i])+' '+np.str(ave_a_home[i])+' '+np.str(ave_b_home[i])+' '+np.str(ave_a_away[i])+' '+np.str(ave_b_away[i])+'\n')
f.close()
#!!!!!!!!!!end of plotting aberage alpha and beta values by team



   
ntrain = np.shape(teamid)[0]

#histogram of number of home and away games
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(n_home,bins=50,histtype='step',color='r',label='home games')
ax1.hist(n_away,bins=50,histtype='step',color='b',label='away games')
ax1.set_xlabel('number of games')
ax1.set_ylabel('frequency')
plt.legend()
plt.savefig('fig_homeandawayhist.pdf')




#load the test data and identify any teams present in the test sample and not training sample 

csvFile = 'test.csv'
csvRead = csv.reader(open(csvFile))
next(csvRead)
teamid_home = []
teamid_away = []
ngame_home = []
ngame_away = []
league_id = []


for game in csvRead:
 league_id.append(np.float(game[1]))
 teamid_home.append(np.float(game[3]))
 teamid_away.append(np.float(game[4]))

teamid_home = np.array(teamid_home)
teamid_away = np.array(teamid_away)
league_id = np.array(league_id)
ngame = np.shape(teamid_home)[0]
league_id_u   = np.unique(league_id)
nleague = np.shape(league_id_u)[0]
teamid_home_u = np.unique(teamid_home)
teamid_away_u = np.unique(teamid_away)

#lid = []
#for tid_h in teamid_home_u:
# idl = np.where(teamid == tid_h)[0] 
# lid.append(league_id[idl])
#lid = np.array(lid)


teamid_u = np.unique(np.concatenate([teamid_home,teamid_away]))
nu = np.shape(teamid_u)[0]
nhome_u = np.zeros(nu)
naway_u = np.zeros(nu)

nhu = np.shape(nhome_u)[0]
nau = np.shape(naway_u)[0]

naway_train_u = np.zeros(nu)
nhome_train_u = np.zeros(nu)
for i in range(nu):
 tidnow = teamid_u[i]
 for i2 in range(ntrain):
  if (tidnow == teamid[i2]):
   nhome_train_u[i] = nhome_train_u[i] + n_home[i]
   naway_train_u[i] = naway_train_u[i] + n_away[i]
 
 for i2 in range(ngame):
  if (tidnow == teamid_home[i2]):
   nhome_u[i] = nhome_u[i] + 1
  if (tidnow == teamid_home[i2]):
   naway_u[i] = naway_u[i] + 1

  
#for each team in test data have
#nhome_u, naway_u number of home and away games played in test data
#nhome_train_u, naway_train_u number of home and away games played in training data

for il in range(nleague):
 lnow   = league_id_u[il]
 idxnow = np.where(league_id == lnow)[0]#the games in a certain league id now need the number of home goals for all teams in that league
 
 idhome_u = np.unique(teamid_home[idxnow])
 idaway_u = np.unique(teamid_away[idxnow])
 
 #identify indicees where the unique team id corresponds to the league we are plotting now
 idxhome_u = np.where(np.in1d(idhome_u, teamid_u))[0]
 idxaway_u = np.where(np.in1d(idaway_u, teamid_u))[0]
 
 
 nnow_home   = np.shape(idhome_u)[0]
 nnow_away   = np.shape(idaway_u)[0]
 xnow_home   = np.arange(nnow_home)
 xnow_away   = np.arange(nnow_away)
 
 
 #idxhome_train = np.where(nhome_train_u
 ynow_home_train   = nhome_train_u[idxhome_u]
 ynow_away_train   = naway_train_u[idxaway_u] 
 ynow_home_test    = nhome_u[idxhome_u]
 ynow_away_test    = naway_u[idxaway_u]
 
 
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 x = np.arange(nu)
 bar_width = 0.35
 opacity = 0.4
 ax1.bar(xnow_home, ynow_home_train, bar_width,
                 alpha=opacity, color='b',
                 label='Training')
 ax2=ax1.twinx()
 ax2.bar(xnow_home+bar_width, ynow_home_test, bar_width,
                 alpha=opacity, color='r',
                 label='Test')
 ax1.set_ylabel('Number of home corners (test data)')
 ax2.yaxis.tick_right()
 ax2.set_ylabel('Number of away corners (test data)')
 ax1.set_xlabel('team id')
 h1, l1 = ax1.get_legend_handles_labels()
 h2, l2 = ax2.get_legend_handles_labels()
 ax1.legend(h1+h2, l1+l2, loc=2)
 ax1.set_xticks(x)
 ax1.set_xticklabels([np.str(teamid_u[i]) for i in range(nu)])
 plt.savefig('fig_homebar_'+np.str(il)+'.pdf')
 
 
 
 
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 ax1.bar(xnow_away, ynow_away_train, bar_width,
                 alpha=opacity, color='b',
                 label='Training')
 ax2=ax1.twinx()
 ax2.bar(xnow_away+bar_width, ynow_away_test, bar_width,
                 alpha=opacity, color='r',
                 label='Test')
 ax1.set_ylabel('Number of home corners (test data)')
 ax2.yaxis.tick_right()
 ax2.set_ylabel('Number of away corners (test data)')
 ax1.set_xlabel('team id')
 h1, l1 = ax1.get_legend_handles_labels()
 h2, l2 = ax2.get_legend_handles_labels()
 ax1.legend(h1+h2, l1+l2, loc=2)
 ax1.set_xticks(x)
 ax1.set_xticklabels([np.str(teamid_u[i]) for i in range(nu)])
 plt.savefig('fig_awaybar_'+np.str(il)+'.pdf')





#plot example posterior probability distribution
psave = []
hc = []
ac = []
with open('poisson.txt') as f:
	probsum = 0.0
	for line in f:
		
		home_corners_m = int(line.split(' = ')[0][4])
		away_corners_m = int(line.split(' = ')[0][5])
		prob = float(line.split(' = ')[1])
		psave.append(prob)
		hc.append(home_corners_m)
		ac.append(away_corners_m)
psave = np.array(psave)
hc = np.array(hc)
ac = np.array(ac)


plt.clf()
N = int(len(psave)**.5)
z = psave.reshape(N, N)
plt.imshow(z, extent=(np.amin(hc), np.amax(hc), np.amin(ac), np.amax(ac)), aspect = 'auto')
cbar = plt.colorbar()
plt.xlabel(r'$i$ Home Corners')
plt.ylabel(r'$j$ Away Corners')
cbar.set_label(r'2D probability distribution $P_{ij}$')  
plt.savefig('fig_eg_posterior.pdf')
#nh = np.shape(teamid_home)[0]
#na = np.shape(teamid_away)[0]
#nhu = np.shape(teamid_home_u)[0]
#nau = np.shape(teamid_away_u)[0]
#
#
##count the number of home and away games each team plays in the test data set
#nhome_teamid_u = np.zeros(nhu)
#naway_teamid_u = np.zeros(nau)
#nhome_train = []
#naway_train = []
#for i in range(nhu):
# tidnow = teamid_home_u[i]
# for i2 in range(nh):
#  if (tidnow == teamid_home[i2]):
#   nhome_teamid_u[i] = nhome_teamid_u[i] + 1
# 
#  nhtnow = 0
# try:
#  idtrain = np.where(teamid == tidnow)[0]
#  nhtnow = n_home[idtrain]
# except:
#  print 'team id',tidnow,' is not present in the training set'
# nhome_train.append(nhtnow)
#   
#for i in range(nau):
# tidnow = teamid_away_u[i]
# for i2 in range(na):
#  if (tidnow == teamid_away[i2]):
#   naway_teamid_u[i] = naway_teamid_u[i] + 1
#  
#  natnow = 0
# try:
#  idtrain = np.where(teamid == tidnow)[0]
#  natnow = n_away[idtrain]
# except:
#  print 'team id',tidnow,' is not present in the training set'
# naway_train.append(natnow)    
# 
# 
#nhome_train, naway_train, nhome_teamid_u,naway_teamid_u,teamid_home_u,teamid_away_u 
#compare the number of home and away games each team played in the test data set 
#with the number of home and away games played in the training set

#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(nhome_train,color='r',label='training')
#ax2=ax1.twinx()
#ax2.plot(nhome_teamid_u,color='b',label='test')
#ax2.yaxis.tick_right()
#ax2.set_ylabel('test games')
#ax1.set_ylabel('training games')
#
#plt.savefig('fig_home_testtrain.pdf')
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(naway_train,color='r',label='training')
#ax2=ax1.twinx()
#ax2.plot(naway_teamid_u,color='b',label='test')
#ax2.yaxis.tick_right()
#ax2.set_ylabel('test games')
#ax1.set_ylabel('training games')
#plt.savefig('fig_away_testtrain.pdf')
# 
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
#from collections import namedtuple
##n_groups = 5
##means_men = (20, 35, 30, 35, 27)
##std_men = (2, 3, 4, 1, 2)
##means_women = (25, 32, 34, 20, 25)
##std_women = (3, 5, 2, 3, 3)
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
##fig, ax = plt.subplots()
##index = np.arange(n_groups)
#bar_width = 0.35
#opacity = 0.4
##error_config = {'ecolor': '0.3'}
#rects1 = ax1.bar(index, means_men, bar_width,
#                alpha=opacity, color='b',
#                label='Training')
#index = np.arange(nau)
#ax2 = ax1.twinx()
#rects2 = ax2.bar(index + bar_width, means_women, bar_width,
#                alpha=opacity, color='r',
#                label='Test')
##ax.set_xlabel('Group')
##ax.set_ylabel('Scores')
##ax.set_title('Scores by group and gender')
##ax.set_xticks(index + bar_width / 2)
##ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
##ax.legend()
##fig.tight_layout()
##plt.show()
## 
# 
# 
# 
# 