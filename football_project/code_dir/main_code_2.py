#load data
import numpy as np
import posixpath
import pandas as pd
import scipy.optimize as op





root_url = 'http://www.football-data.co.uk/mmz4281/'
season = '1617'
league = 'E0' # Premiere League
data_file_suffix = '.csv'
remote_file = posixpath.join(root_url, season,
                             '{0}{1}'.format(league, data_file_suffix))
data = pd.read_csv(remote_file)
print(data.shape)




#function to correct for scores 0-0 and 1-1 being more common
def tau_function(x, y, lambdaa, mu, rho):
    if x == 0 and y == 0:
        tau = 1 - lambdaa * mu * rho
    elif x == 0 and y == 1:
        tau = 1 + lambdaa * rho
    elif x == 1 and y == 0:
        tau = 1 + mu * rho
    elif x == 1 and y == 1:
        tau = 1 - rho
    else:
        tau = 1
    return tau





def get_time_weights(dates, epsilon):
    delta_days = [(max(dates) - d).days for d in dates]
    # future games not relevant
    return list(map(lambda x: 0 if x < 0 else np.exp(-1 * epsilon * x), delta_days))






def time_ln_likelihood(values):
   return sum([(value['time_weights'] *
                (np.log(tau_function(value['home_goals'], value['away_goals'],
                                     value['lambda'], value['mu'], value['rho'])) +
                 (-value['lambda']) + value['home_goals'] * np.log(value['lambda']) +
                 (-value['mu']) + value['away_goals'] * np.log(value['mu'])))
               for value in values])



def norm_alphas(params, number_of_teams):
   return sum(params[:number_of_teams]) / number_of_teams - 1




minimize_result = op.minimize(fun=lambda *args: -time_ln_likelihood(*args),
                             x0=x0,
                             args=(input_data_frame['HomeId'].tolist(),  # home teams
                                   input_data_frame['AwayId'].tolist(),  # away teams
                                   input_data_frame['FTHG'].tolist(),  # home goals
                                   input_data_frame['FTAG'].tolist(),  # away goals
                                   number_of_teams,
                                   get_time_weights(input_data_frame['Date'].tolist())),
                             constraints=({'type': 'eq', 'fun': lambda *args: norm_alphas(*args), 'args': [number_of_teams]}))



win_probability = sum(sum(np.tril(game_probabilities, -1)))  # triangle-lower for home win
draw_probability = game_probabilities.trace()  # diagonal for draw
loss_probability = sum(sum(np.triu(game_probabilities, 1)))  # triangle-upper for home loss





MODEL_FITTING_GRID = {"method":['Dixon-Coles', 'Maher'],
                     "epsilon": append(arange(.0005, .00225, .00025).tolist(), [0.0])}
BETTING_GRID = {"betting_strategy": ['fixed_bet', 'Kelly', 'variance_adjusted'],
               "bet_threshold": arange(1.05, 1.55, .005)}
