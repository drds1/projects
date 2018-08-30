import csv, math, ast 
import numpy as np
import matplotlib.pylab as plt
import json

def poisson(actual, mean):
    return math.pow(mean, actual) * math.exp(-mean) / math.factorial(actual)






league_id_train_u = []
ave_a_home = []
ave_b_home = []
ave_a_away = []
ave_b_away = []
try:
 #!!!!!!!!!!!load the average alpha, beta values averaged for each league
 with open('alpha_beta_league_stats.txt') as f:
     content = f.readlines()
 # you may also want to remove whitespace characters like `\n` at the end of each line
 content = [x.strip() for x in content]
 for cnow in content:
  l,ah,bh,aa,ba = [float(i) for i in cnow.split()]
  league_id_train_u.append(l)
  ave_a_home.append(ah)
  ave_b_home.append(bh)
  ave_a_away.append(aa)
  ave_b_away.append(ba)
 f.close()
 
 league_id_train = np.array(league_id_train)
 ave_a_home = np.array(ave_a_home)
 ave_b_home = np.array(ave_b_home)
 ave_a_away = np.array(ave_a_away)
 ave_b_away = np.array(ave_b_away)
 league_ave_search = 1
 #!!!!!!!!!!!
except:
 league_ave_search = 0
 pass


league_ave_search = 0

csvFile = 'train.csv'
tau = 1.0
team_list = []
tvsum = []

k = open('team_list.txt', 'w')
k.write("""{
""")

csvRead = csv.reader(open(csvFile))
next(csvRead)

for row in csvRead:
	if row[3] not in team_list:
		team_list.append(row[3])
	if row[4] not in team_list:
		team_list.append(row[4])

team_list.sort()

for team in team_list:
	k.write("""	'%s': {'home_corners': 0, 'away_corners': 0, 'home_conceded': 0, 'away_conceded': 0, 'home_games': 0, 'away_games': 0, 'alpha_h': 0, 'beta_h': 0, 'alpha_a': 0, 'beta_a': 0},
""" % (team))

k.write("}")
k.close()

s = open('team_list.txt', 'r').read()
dict = ast.literal_eval(s)

GAMES_PLAYED = 0
WEEKS_WAIT = 4
TOTAL_VALUE = 0

csvRead = csv.reader(open(csvFile))
next(csvRead)

idx = 0
for game in csvRead:
    #skip null results
	if (game[5] == 'NULL'):
	 print 'game',idx,' null result... skipping'
	 continue
	 
	if (np.mod(idx,1000.0) == 0):
	 print idx,' entries read...' 
	
	home_team = game[3]
	away_team = game[4]
	league_id_now = np.float(game[1])
	idxref = np.where(league_id_train_u == league_id_now)[0]
    

    
	home_corners = int(game[7])
	away_corners = int(game[8])

	home_win_prob = 0
	draw_win_prob = 0
	away_win_prob = 0
	
	curr_home_corners = 0
	curr_away_corners = 0
	avg_home_corners = 1
	avg_away_corners = 1
	
	team_bet = ''
	ev_bet = ''
	
	# GETTING UPDATED VARIABLES
	for key, value in dict.items():
		curr_home_corners += dict[key]['home_corners']
		curr_away_corners += dict[key]['away_corners']
		
		if GAMES_PLAYED > (WEEKS_WAIT * 10):
			avg_home_corners = np.float(curr_home_corners) / (GAMES_PLAYED)
			avg_away_corners = np.float(curr_away_corners) / (GAMES_PLAYED)
	
	
	# CALCULATING FACTORS
	if GAMES_PLAYED > (WEEKS_WAIT * 10):
		home_team_a = (dict[home_team]['alpha_h'] + dict[home_team]['alpha_a']) / 2
		away_team_a = (dict[away_team]['alpha_h'] + dict[away_team]['alpha_a']) / 2
		
		home_team_d = (dict[home_team]['beta_h'] + dict[home_team]['beta_a']) / 2
		away_team_d = (dict[away_team]['beta_h'] + dict[away_team]['beta_a']) / 2
		
		home_team_exp = avg_home_corners * home_team_a * away_team_d
		away_team_exp = avg_away_corners * away_team_a * home_team_d
	
	
	# RUNNING POISSON	
		l = open('poisson.txt', 'w')
		
		for i in range(10):
			for j in range(10):
				prob = tau * poisson(i, home_team_exp) * poisson(j, away_team_exp)
				l.write("Prob%s%s = %s\n" % (i, j, prob))
		
		l.close()
		
		with open('poisson.txt') as f:
			probsum = 0.0
			for line in f:
				
				home_corners_m = int(line.split(' = ')[0][4])
				away_corners_m = int(line.split(' = ')[0][5])
				
				prob = float(line.split(' = ')[1])
				probsum = probsum + prob
				if home_corners_m > away_corners_m:
					home_win_prob += prob
				elif home_corners_m == away_corners_m:
					draw_win_prob += prob
				elif home_corners_m < away_corners_m:
					away_win_prob += prob

	#CALCULATE VALUE
		#bet365odds_h, bet365odds_d, bet365odds_a = float(game[23]), float(game[24]), float(game[25])
		#
		#ev_h = (home_win_prob/probsum * (bet365odds_h - 1)) - (1 - home_win_prob/probsum)
		#ev_d = (draw_win_prob/probsum * (bet365odds_d - 1)) - (1 - draw_win_prob/probsum)
		#ev_a = (away_win_prob/probsum * (bet365odds_a - 1)) - (1 - away_win_prob/probsum)
		#
		#highestEV = max(ev_h, ev_d, ev_a)
		#
		#if (ev_h == highestEV) and (ev_h > 0):
		#	team_bet = home_team
		#	ev_bet = ev_h
		#	if home_goals > away_goals:
		#		TOTAL_VALUE += (bet365odds_h - 1)
		#	else:
		#		TOTAL_VALUE -= 1
		#		
		#elif (ev_d == highestEV) and (ev_d > 0):
		#	team_bet = 'Draw'
		#	ev_bet = ev_d
		#	if home_goals == away_goals:
		#		TOTAL_VALUE += (bet365odds_d - 1)
		#	else:
		#		TOTAL_VALUE -= 1
		#elif (ev_a == highestEV) and (ev_a > 0):
		#	team_bet = away_team
		#	ev_bet = ev_a
		#	if home_goals < away_goals:
		#		TOTAL_VALUE += (bet365odds_a - 1)
		#	else:
		#		TOTAL_VALUE -= 1
		#
		#
		#tvsum.append(TOTAL_VALUE)
		#if (team_bet != '') and (ev_bet != ''):
		#	print ("Bet on '%s' (EV = %s)" % (team_bet, ev_bet))	
		#	print (TOTAL_VALUE)
		
	# UPDATE VARIABLES AFTER MATCH HAS BEEN PLAYED
	dict[home_team]['home_corners'] += home_corners
	dict[home_team]['home_conceded'] += away_corners
	dict[home_team]['home_games'] += 1
	
	dict[away_team]['away_corners'] += away_corners
	dict[away_team]['away_conceded'] += home_corners
	dict[away_team]['away_games'] += 1
	
	GAMES_PLAYED += 1
	
	# CREATE FACTORS
	if GAMES_PLAYED > (WEEKS_WAIT * 10):
		for key, value in dict.items():
			
			#if the team hasn't played any games yet just set the modifier to 1
			#i.e follow the expected result
			hg = np.float(dict[key]['home_games'])
			ag = np.float(dict[key]['away_games'])
			if (hg >0):
			 alpha_h = (dict[key]['home_corners'] / hg) / avg_home_corners
			 beta_h = (dict[key]['home_conceded'] / hg) / avg_away_corners
			else:
			 if (league_ave_search == 0):
			  alpha_h = 1.0
			  beta_h = 1.0
			 else:
			  idl = np.where(league_id_train == league_id_now)[0]
			  if (len(idl) == 0):
			   alpha_h = 1.0
			   beta_h = 1.0
			  else:
			   alpha_h = ave_a_home[idl[0]]
			   beta_h = ave_b_home[idl[0]]
			 
			if (ag > 0):
			 alpha_a = (dict[key]['away_corners'] / ag) / avg_away_corners
			 beta_a = (dict[key]['away_conceded'] / ag) / avg_home_corners
			else:
			 if (league_ave_search == 0):
			  alpha_a = 1.0
			  beta_a = 1.0
			 else:
			  idl = np.where(league_id_train == league_id_now)[0]
			  if (len(idl) == 0):
			   alpha_a = 1.0
			   beta_a = 1.0
			  else:
			   alpha_a = ave_a_away[idl[0]]
			   beta_a = ave_b_away[idl[0]]

			

			dict[key]['alpha_h'] = alpha_h
			dict[key]['beta_h'] = beta_h
			dict[key]['alpha_a'] = alpha_a
			dict[key]['beta_a'] = beta_a
			
	idx = idx + 1
	#print 'loaded game',idx+1

#dump the average number of home and away corners
np.savetxt('avg_home_away.txt',np.array([avg_home_corners,avg_away_corners]))

#dump the dictionary trained on the training data set
with open('dict_dump.txt', 'w') as file:
     file.write(json.dumps(dict))			
file.close()		
	
	



#plot results
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(tvsum)
#xl = list(ax1.get_xlim())
#ax1.plot(xl,[0,0],ls='--',color='k')
#plt.savefig('fig_profit.png')			
			