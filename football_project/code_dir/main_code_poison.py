import csv, math, ast 
import numpy as np
import matplotlib.pylab as plt

def poisson(actual, mean):
    return math.pow(mean, actual) * math.exp(-mean) / math.factorial(actual)

csvFile = '20152016.csv'
tau = 1.0
team_list = []
tvsum = []

k = open('team_list.txt', 'w')
k.write("""{
""")

csvRead = csv.reader(open(csvFile))
next(csvRead)

for row in csvRead:
	if row[2] not in team_list:
		team_list.append(row[2])
	if row[3] not in team_list:
		team_list.append(row[3])

team_list.sort()

for team in team_list:
	k.write("""	'%s': {'home_goals': 0, 'away_goals': 0, 'home_conceded': 0, 'away_conceded': 0, 'home_games': 0, 'away_games': 0, 'alpha_h': 0, 'beta_h': 0, 'alpha_a': 0, 'beta_a': 0},
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

for game in csvRead:
	home_team = game[2]
	away_team = game[3]

	home_goals = int(game[4])
	away_goals = int(game[5])

	home_win_prob = 0
	draw_win_prob = 0
	away_win_prob = 0
	
	curr_home_goals = 0
	curr_away_goals = 0
	avg_home_goals = 1
	avg_away_goals = 1
	
	team_bet = ''
	ev_bet = ''
	
	# GETTING UPDATED VARIABLES
	for key, value in dict.items():
		curr_home_goals += dict[key]['home_goals']
		curr_away_goals += dict[key]['away_goals']
		
		if GAMES_PLAYED > (WEEKS_WAIT * 10):
			avg_home_goals = curr_home_goals / (GAMES_PLAYED)
			avg_away_goals = curr_away_goals / (GAMES_PLAYED)
	
	
	# CALCULATING FACTORS
	if GAMES_PLAYED > (WEEKS_WAIT * 10):
		home_team_a = (dict[home_team]['alpha_h'] + dict[home_team]['alpha_a']) / 2
		away_team_a = (dict[away_team]['alpha_h'] + dict[away_team]['alpha_a']) / 2
		
		home_team_d = (dict[home_team]['beta_h'] + dict[home_team]['beta_a']) / 2
		away_team_d = (dict[away_team]['beta_h'] + dict[away_team]['beta_a']) / 2
		
		home_team_exp = avg_home_goals * home_team_a * away_team_d
		away_team_exp = avg_away_goals * away_team_a * home_team_d
	
	
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
				
				home_goals_m = int(line.split(' = ')[0][4])
				away_goals_m = int(line.split(' = ')[0][5])
				
				prob = float(line.split(' = ')[1])
				probsum = probsum + prob
				if home_goals_m > away_goals_m:
					home_win_prob += prob
				elif home_goals_m == away_goals_m:
					draw_win_prob += prob
				elif home_goals_m < away_goals_m:
					away_win_prob += prob

	#CALCULATE VALUE
		bet365odds_h, bet365odds_d, bet365odds_a = float(game[23]), float(game[24]), float(game[25])
		
		ev_h = (home_win_prob/probsum * (bet365odds_h - 1)) - (1 - home_win_prob/probsum)
		ev_d = (draw_win_prob/probsum * (bet365odds_d - 1)) - (1 - draw_win_prob/probsum)
		ev_a = (away_win_prob/probsum * (bet365odds_a - 1)) - (1 - away_win_prob/probsum)
		
		highestEV = max(ev_h, ev_d, ev_a)
		
		if (ev_h == highestEV) and (ev_h > 0):
			team_bet = home_team
			ev_bet = ev_h
			if home_goals > away_goals:
				TOTAL_VALUE += (bet365odds_h - 1)
			else:
				TOTAL_VALUE -= 1
				
		elif (ev_d == highestEV) and (ev_d > 0):
			team_bet = 'Draw'
			ev_bet = ev_d
			if home_goals == away_goals:
				TOTAL_VALUE += (bet365odds_d - 1)
			else:
				TOTAL_VALUE -= 1
		elif (ev_a == highestEV) and (ev_a > 0):
			team_bet = away_team
			ev_bet = ev_a
			if home_goals < away_goals:
				TOTAL_VALUE += (bet365odds_a - 1)
			else:
				TOTAL_VALUE -= 1
		
		
		tvsum.append(TOTAL_VALUE)
		if (team_bet != '') and (ev_bet != ''):
			print ("Bet on '%s' (EV = %s)" % (team_bet, ev_bet))	
			print (TOTAL_VALUE)
		
	# UPDATE VARIABLES AFTER MATCH HAS BEEN PLAYED
	dict[home_team]['home_goals'] += home_goals
	dict[home_team]['home_conceded'] += away_goals
	dict[home_team]['home_games'] += 1
	
	dict[away_team]['away_goals'] += away_goals
	dict[away_team]['away_conceded'] += home_goals
	dict[away_team]['away_games'] += 1
	
	GAMES_PLAYED += 1
	
	# CREATE FACTORS
	if GAMES_PLAYED > (WEEKS_WAIT * 10):
		for key, value in dict.items():
			alpha_h = (dict[key]['home_goals'] / dict[key]['home_games']) / avg_home_goals
			beta_h = (dict[key]['home_conceded'] / dict[key]['home_games']) / avg_away_goals

			alpha_a = (dict[key]['away_goals'] / dict[key]['away_games']) / avg_away_goals
			beta_a = (dict[key]['away_conceded'] / dict[key]['away_games']) / avg_home_goals

			dict[key]['alpha_h'] = alpha_h
			dict[key]['beta_h'] = beta_h
			dict[key]['alpha_a'] = alpha_a
			dict[key]['beta_a'] = beta_a
			
			
			
#plot results
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(tvsum)
xl = list(ax1.get_xlim())
ax1.plot(xl,[0,0],ls='--',color='k')
plt.savefig('fig_profit.png')			
			