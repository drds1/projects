from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv
import pandas as pd
import numpy as np

# one-step sarima forecast
def varma_forecast(history, config):
    order, trend = config
    # define model
    model = VARMAX(history, order=order, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = varma_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# create a set of sarima configs to try
def varma_configs(p_params = [0, 1, 2],
                   q_params = [0, 1, 2],
                   t_params = ['n', 'c', 't', 'ct']):
    models = list()
    # create config instances
    for p in p_params:
        for q in q_params:
            for t in t_params:
                cfg = [(p, q), t]
                models.append(cfg)
    return models



class varma_CV:
    '''
    perform a cross validated test of the sarima modelling
    '''

    def __init__(self):
        self.data = 'monthly-car-sales.csv'
        self.n_test = 12
        self.p_params = [0, 1, 2]
        self.q_params = [0, 1, 2]
        self.t_params = ['n', 'c', 't', 'ct']


    def run_CV(self):
        '''
        run the cross validation with the inputted configs
        :return:
        '''
        if type(self.data) is str:
            series = read_csv(self.data, header=0, index_col=0)
            self.data = series
            data = series.values
        else:
            data = self.data.values


        cfg_list = varma_configs(p_params=self.p_params,
                                  q_params=self.q_params,
                                  t_params=self.t_params)
        # grid search
        s = grid_search(data, cfg_list, self.n_test)
        score_df = {'p': [], 'q': [], 'trend': []}
        for i in range(len(s)):
            pdq_PDQS = [int(ss) for ss in s[i][0] if ss.isdigit()]
            trend_now = s[i][0].split("'")[1]
            score_df['p'].append(pdq_PDQS[0])
            score_df['q'].append(pdq_PDQS[1])
            score_df['trend'].append(trend_now)
        self.scores = pd.DataFrame(score_df)
        print('done')
        # list top 3 configs
        for cfg, error in s[:3]:
            print(cfg, error)





if __name__ == '__main__':
    # load dataset
    data = pd.read_csv('monthly-car-sales.csv')
    data['test 1'] = np.random.randn(len(data))
    data['test 2'] = np.random.randn(len(data))
    data.set_index('Month',inplace=True)
    X = varma_CV()
    X.data = data
    X.run_CV()