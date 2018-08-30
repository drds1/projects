import csv, math, ast 
import numpy as np
import matplotlib.pylab as plt
import json

def poisson(actual, mean):
    return math.pow(mean, actual) * math.exp(-mean) / math.factorial(actual)






opfile = 'test_op.csv'
ophead = ['MatchId','LeagueId','Date','HomeTeamId','AwayTeamId','Line','Over','Under','','P(Under)','P(At)','P(Over)','Bet (U/O)','Stake']


csvFile = 'test.csv'
tau = 1.0
team_list = []
tvsum = []

#k = open('team_list.txt', 'w')
#k.write("""{
#""")

#!!!!!!!!!!!load the average alpha, beta values averaged for each league
league_id_train_u = []
ave_a_home = []
ave_b_home = []
ave_a_away = []
ave_b_away = []
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
#!!!!!!!!!!!





csvRead = csv.reader(open(csvFile))
next(csvRead)

#for row in csvRead:
#	if row[3] not in team_list:
#		team_list.append(row[3])
#	if row[4] not in team_list:
#		team_list.append(row[4])
#
#team_list.sort()

#for team in team_list:
#	k.write("""	'%s': {'home_corners': 0, 'away_corners': 0, 'home_conceded': 0, 'away_conceded': 0, 'home_games': 0, 'away_games': 0, 'alpha_h': 0, 'beta_h': 0, 'alpha_a': 0, 'beta_a': 0},
#""" % (team))
#
#k.write("}")
#k.close()

s = open('dict_dump.txt', 'r').read()
dict = ast.literal_eval(s)

GAMES_PLAYED = 0
WEEKS_WAIT = 4
TOTAL_VALUE = 0


pat_save = []
pb_save = []
pa_save = []
bet_save = []
stake_save = []
row_save = []


csvRead = csv.reader(open(csvFile))
next(csvRead)

for game in csvRead:
    #skip null results
	if (game[5] == 'NULL'):
	 continue
	
	home_team = game[3]
	away_team = game[4]
	corner_line = np.float(game[5])
	#home_corners = int(game[7])
	#away_corners = int(game[8])
#
	#home_win_prob = 0
	#draw_win_prob = 0
	#away_win_prob = 0
	#
	#curr_home_corners = 0
	#curr_away_corners = 0
	#avg_home_corners = 1
	#avg_away_corners = 1
	#
	#team_bet = ''
	#ev_bet = ''
	#
	## GETTING UPDATED VARIABLES
	#for key, value in dict.items():
	#	curr_home_corners += dict[key]['home_corners']
	#	curr_away_corners += dict[key]['away_corners']
	#	
	#	if GAMES_PLAYED > (WEEKS_WAIT * 10):
	#		avg_home_corners = np.float(curr_home_corners) / (GAMES_PLAYED)
	#		avg_away_corners = np.float(curr_away_corners) / (GAMES_PLAYED)
	#
	#
	
	#above only for training sample
	#just need average number of home and away corners and the laoded dictionary from the 
	#training sample
	avg_home_corners,avg_away_corners = np.loadtxt('avg_home_away.txt')
	
	line,betodds_o,betodds_u = np.float(game[5]),np.float(game[6]),np.float(game[7])
	
	# CALCULATING FACTORS
	#if GAMES_PLAYED > (WEEKS_WAIT * 10):
	home_team_a = (dict[home_team]['alpha_h'] + dict[home_team]['alpha_a']) / 2
	away_team_a = (dict[away_team]['alpha_h'] + dict[away_team]['alpha_a']) / 2
	
	home_team_d = (dict[home_team]['beta_h'] + dict[home_team]['beta_a']) / 2
	away_team_d = (dict[away_team]['beta_h'] + dict[away_team]['beta_a']) / 2
	
	home_team_exp = avg_home_corners * home_team_a * away_team_d
	away_team_exp = avg_away_corners * away_team_a * home_team_d
	
	
	
	overline_prob = 0.0
	underline_prob = 0.0
	atline_prob = 0.0
	
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
			
			tot_corners = home_corners_m + away_corners_m
			if (tot_corners > corner_line):
			 overline_prob += prob
			elif (tot_corners < corner_line):
			 underline_prob +=prob
			elif (tot_corners == corner_line):
			 atline_prob += prob
			
			#if home_corners_m > away_corners_m:
			#	home_win_prob += prob
			#elif home_corners_m == away_corners_m:
			#	draw_win_prob += prob
			#elif home_corners_m < away_corners_m:
			#	away_win_prob += prob
#
	#CULATE VALUE
	#bet365odds_h, bet365odds_d, bet365odds_a = float(game[23]), float(game[24]), float(game[25])
	
	
	#expected return on an overline bet
	# E_o = int stake * P_o * (bet_o-1)    -    stake * P_u
	# E_u = int stake * P_u * (bet_u-1)    -    stake * P_o
	
	ev_o = (overline_prob/probsum * (betodds_o - 1)) - (1 - overline_prob/probsum)
	#ev_at = (atline_prob/probsum * (bet365odds_d - 1)) - (1 - atline_prob/probsum)
	ev_u = (underline_prob/probsum * (betodds_u - 1)) - (1 - underline_prob/probsum)
	
	highestEV = max(ev_o, ev_u)#ev_at
	
	print 'P(under),P(at),P(over)',underline_prob/probsum,atline_prob/probsum,overline_prob/probsum
	team_bet = 'no bet'
	stake = 0
	if (ev_o == highestEV) and (ev_o > 0):
		team_bet = 'O'
		ev_bet = ev_o
		stake = 1
		
	elif (ev_u == highestEV) and (ev_u > 0):
		team_bet = 'U'
		ev_bet = ev_u	
		stake = 1
		

	#tvsum.append(TOTAL_VALUE)

	if (team_bet != '') and (highestEV > 0):
		print ("Bet on '%s' (EV = %s)" % (team_bet,highestEV))
	
	game[9] = underline_prob/probsum
	game[10] = atline_prob/probsum
	game[11] = overline_prob/probsum
	game[12] = team_bet
	game[13] = stake
	row_save.append(game)
	pat_save.append(atline_prob/probsum)
	pb_save.append(underline_prob/probsum)
	pa_save.append(overline_prob/probsum)
	bet_save.append(team_bet)
	stake_save.append(stake)
	
	#!!!! NO updating dictionary for testing dataset!


#save opfile
f = open(opfile,'w')
f.write(','.join([np.str(gnow) for gnow in ophead])+'\n')
for row in row_save:
 f.write(','.join([np.str(gnow) for gnow in row])+'\n')
f.close()
#plot results
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(tvsum)
#xl = list(ax1.get_xlim())
#ax1.plot(xl,[0,0],ls='--',color='k')
#plt.savefig('fig_profit.png')			
			