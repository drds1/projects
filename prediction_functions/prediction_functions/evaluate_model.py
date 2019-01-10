import numpy as np

def evaluate_model(y_predicted, y_true, verbose = True):
    '''
    Evaluate fit performance

    :param y_predicted: Predicted y values from any time series forecasting code - 1d numpy array
    :param y_true: Actual y value from time series data - 1d numpy array
    :param verbose: If true, print the statistics to screen
    :return: the mean absolute percentage error (mape), rms, up-down percentage agreement, rms/mean*100
    '''
    rms = np.std(y_true - y_predicted)
    mad = np.median(np.abs(y_true - y_predicted))
    dy_predicted = y_predicted[1:] - y_predicted[:-1]
    dy_true      = y_true[1:] - y_true[:-1]
    ny = len(y_true)
    idupup = np.where((dy_predicted > 0) & (dy_true > 0))[0]
    iddowndown = np.where((dy_predicted < 0) & (dy_true < 0))[0]
    id_agree = np.append(idupup,iddowndown)
    percent_agree = 100*np.float(np.shape(id_agree)[0])/(ny-1)
    mape = 100*np.abs( np.mean(y_predicted - y_true)/np.mean(y_true) )
    rms_mean = np.abs( rms/np.mean(y_true)*100 )

    if verbose is True:
        print('evaluating model...')
        print('precentage correct gradient = ',percent_agree)
        print('rms =',rms)
        print('mape = ',mape)
        print('MAD = ', mad)
        print('')
    return(mape,rms,percent_agree,rms_mean,mad)



