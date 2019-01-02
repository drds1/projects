import os
import sys
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.signal as ss

sys.path.append('.')

#from arbitrage_flow_study.conversion_factors import GASOLINE_BARRELS_PER_MT
GALLONS_PER_CBM = 264.17
GALLONS_PER_BBL = 42
BARRELS_PER_CBM = GALLONS_PER_CBM / GALLONS_PER_BBL
GASOLINE_GALLONS_PER_MT = 349.90
GASOLINE_BARRELS_PER_MT = GASOLINE_GALLONS_PER_MT / GALLONS_PER_BBL
GASOLINE_RIN_ETHANOL_CONTENT = 0.1



from prediction_functions.rw_plus_poly import rwp
from prediction_functions.polyfit import fit
from prediction_functions.rw_simple import rw
from prediction_functions.signal_decomp import *
from prediction_functions.ccf_simple import *

path = os.getcwd() + "/resources/"

# path = os.getcwd() + "/arbitrage_flow_study/resources/"
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)
np.warnings.filterwarnings('ignore')


plt.style.use('dark_background')


def r2_score(ytrue,ypred):
    std_res = np.sum((ytrue - ypred)**2)
    ymean   = np.mean(ytrue)
    std_tot = np.sum((ytrue - ymean)**2)
    return(1. - std_res/std_tot)


def calculate_mape(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def dates_to_array(dates, dates_min_in=False):
    if dates_min_in == False:
        dates_min = min(dates)
    else:
        dates_min = dates_min_in
    timedelta_dates = pd.TimedeltaIndex(dates - dates_min)
    days_output = np.array(timedelta_dates.days, dtype='float')
    return days_output


def select_frequencies(ny, periods=[365], upper_period=30):
    # deal with smooth-ness of fit. Reducing 'upper_period'
    # argument will return a better fit but risk overfitting
    if periods is None:
        flo = 1. / ny
        fhi = 1. / upper_period
        frequencies = np.arange(flo, fhi, flo)
    else:
        frequencies = 1. / np.array(periods)
    return (frequencies)


def select_times(dates, predict=0, datesmin_custom = None):
    # convert pandas date time stamp to numpy array of days since start date
    if (type(dates) == np.ndarray):
        return (dates, dates)
    else:
        time = dates_to_array(dates)

        if datesmin_custom is not None:
            datesmin = datesmin_custom
        else:
            datesmin = dates.values[0]

        dates_total = pd.Timestamp(datesmin) + pd.timedelta_range(start='0 day', periods=time[-1] + 1 + predict)
        return (time, dates_total)





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
    dy_predicted = y_predicted[1:] - y_predicted[:1]
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






def predict_seasonality(y, dates=None, periods=None, upper_period=60., predict=10,
                        stats_return=False, iterations=1, interpolate=True,
                        dates_output = None, trend_order = 1,model='fourier'):
    # fit smooth model to time series seasonality
    # and predict 'predict' number of days into the future
    #dates output overrides interpolate argument
    ny = len(y)
    frequencies = select_frequencies(ny, periods=periods, upper_period=upper_period)
    if dates is not None:
        time, dates_total = select_times(dates, predict=predict)
    else:
        time = np.arange(ny)


    if dates_output is not None:
        time_total, crp = select_times(dates_output, predict=0)
    elif interpolate is True:
        time_total = np.arange(time[0], time[-1] + 1 + predict, 1)
    else:
        time_total = np.append(time[:-1], time[-1] + np.arange(predict + 1))

    if model == 'fourier':
        yg_tot, sg_tot, rmd = rwp(time, y, tgrid=time_total, order=trend_order,
                              custom_freqs=frequencies, nits=iterations)
    else:
        yg_tot = np.interp(time_total,time,y)
        sg_tot = yg_tot*0
        rmd = 1.0

    y_predict = yg_tot[-predict:]
    if stats_return is False:
        return (y_predict)
    else:
        return (yg_tot, sg_tot, rmd, time_total)




def model_seasonality(y, dates=None, periods=None, upper_period=60.,
                      interpolate=True):
    # fit smooth model to time series seasonality
    yg_tot = predict_seasonality(y, dates=dates, periods=periods, upper_period=upper_period, predict=0,
                                 interpolate=interpolate)
    return (yg_tot)


def optimize_seasonality(y, dates=None, train_fraction=0.8,optimize_frequencies = False):
    # identify the optimum number of components to fit on model time series
    ny = len(y)
    ntrain = np.int(ny * train_fraction)
    predict = ny - ntrain
    if dates is not None:
        times = select_times(dates)[0]
    else:
        times = np.arange(ny)[0]

    tlo, thi = np.min(times), np.max(times)
    cadence = np.max(times[1:] - times[:-1])

    if optimize_frequencies is True:
        output = prioritize_frequencies(y, dates=None, train_fraction=train_fraction)
        nf = output['use_first_n']
        frequencies = output['frequencies'][:nf]
        r2 = output['r2'][nf]
        rmse = output['rmse'][nf]
    else:
        flo = 1. / (thi - tlo)
        fhi = 0.5 / cadence
        frequencies = np.arange(flo, fhi, flo)
        variance_total = np.var(y)


    # explained variance metric
    variance_components = []
    rmse = []
    period = []
    bic = []
    n_harmonics = 1
    for f in frequencies:
        n_parameters = 2 * n_harmonics + 1
        period.append(1. / f)
        yg_tot = predict_seasonality(y, dates=times, periods=period, predict=0,
                                     interpolate=False)
        variance_current = np.var(y - yg_tot)
        bic.append(ny * np.log(variance_current) + n_parameters * np.log(ny))
        variance_components.append(1. - variance_current / variance_total)

        # mse metric on trained data
        # test the predictive power of each frequency component
        y_predict = predict_seasonality(y[:ntrain], dates=times[:ntrain],
                                        periods=period, interpolate=False, predict=predict)
        y_true = y[ntrain:]
        rmse.append(np.std(y_true - y_predict))
        n_harmonics = n_harmonics + 1

    # return an array of periods, explained_variance, trained_variance, BIC
    output = {'periods': period,
              'explained_variance': variance_components,
              'rmse': rmse,
              'BIC': bic}
    return (output)





def prioritize_frequencies(y, dates=None, train_fraction=0.95):
    # identify the optimum number of components to fit on model time series
    ny = len(y)
    ntrain = np.int(ny * train_fraction)
    predict = ny - ntrain
    if dates is not None:
        times = select_times(dates)[0]
    else:
        times = np.arange(ny)[0]

    tlo, thi = np.min(times), np.max(times)
    cadence = np.max(times[1:] - times[:-1])
    flo = 1.0 / (thi - tlo)
    fhi = 0.5 / cadence
    frequencies = np.arange(flo, fhi, flo)
    nf = len(frequencies)

    variance_total = np.var(y)


    #fit and remove linear trend
    yg_med, yg_lo, yg_hi, cov, cisq, cisq_red, bic, aic, rmd, r2 = fit(times, y, order=1)
    yg_tot = y - yg_med


    tgrid, ygridop, f, ckout, skout, rcoef = rw(times[:ntrain], yg_tot[:ntrain], si=0, tgi=-1, fbreak=-1,
                                                custom_freqs=frequencies, nits=1)


    amplitudes = np.sqrt(skout**2 + ckout**2)

    #sort frequnecies in descending order by amplitude
    idx_sort = np.argsort(amplitudes)[-1::-1]
    f_sort = f[idx_sort]
    sk_sort = skout[idx_sort]
    ck_sort = ckout[idx_sort]
    mse_sort = []
    r2 = []
    ymod = np.zeros(ny)
    for i in range(nf):
        fnow = f_sort[i]
        wnow = 2*np.pi*fnow
        sk = sk_sort[i]
        ck = ck_sort[i]
        ymod = ymod + sk*np.sin(wnow*times) + ck*np.cos(wnow*times)
        variance = np.var(ymod - yg_tot)
        r2now =  1. - variance/variance_total
        r2.append(r2now )
        mse = np.mean((ymod[ntrain:] - yg_tot[ntrain:])**2)
        mse_sort.append(mse)

        plt.clf()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(times,ymod,color='r')
        ax1.plot(times[ntrain:],ymod[ntrain:],color='k',ls='--')
        ax1.plot(times,yg_tot,color='b')
        ax1.set_title(' harmonic '+ np.str(1./fnow)+' days\n'
                                                    'r2 = '+np.str(np.round(r2now,2))+'\n'
                      'mse = '+np.str(np.round(mse)))
        plt.tight_layout()
        plt.show()

    #compute the difference in mse (the harmonic that reduces this th emost should be included
    # and harmonics that increase this should be excluded)
    mse_sort = np.array(mse_sort)
    idsort = np.array(np.append(np.zeros(1),np.argsort(mse_sort[1:] - mse_sort[:-1]) + 1),dtype='int')
    mse_sort = mse_sort[idsort]
    f_sort = np.array(f_sort)[idsort]
    amplitudes = np.array(amplitudes)[idsort]
    r2 = np.array(r2)[idsort]
    idx_stop = np.argmin(mse_sort)


    # return an array of periods, explained_variance, trained_variance, BIC
    output = {'frequencies': f_sort,
              'rmse': mse_sort,
              'amplitudes': amplitudes[idx_sort],
              'r2': np.array(r2),
              'use_first_n': idx_stop}

    return(output)






# lag should be list of number of components with -ve numbers indicating by how much each constituent signal
# leads the flow
#can either
def combine_signal_lag(x, x_components, dates_x, lag=None, predict=0,diagnostic = False,
                       periods = None, upper_period = 30):
    '''
    produces time series forecast based on historic time series behaviour and
    other user specified component. User specified components (e.g ARB) only add
    value to the forecast if they lead the driving (e.g flows) time series.

    :param x: the flows (or driving time series) - 1d numpy array (npoints)
    :param x_components: the components to test against the driver
        for correlation and "forecastability" - 2d numpy array (npoints,ncomponents)
    :param dates_x: a dataframe of date stamps for the x and xcomponents data
        (must be on the same time axis before calling this function)
    :param lag: if present, forces the components to lead (if -ve) or lag (if +ve) the driving
        time series. Only useable for a forecast if lag is -ve (e.g the time series leads.
        - list(ncomponents)
        If not entered, function uses prediction_functions.ccf_simple to find the best fitting lag
    :param predict: function returns "predict" number of days extra as a forecast.
        Can specify predict = 0 and manually add extra dates to the dates_x input for
        greater control (to specify irregular forecast intervals).
    :param diagnostic: Plots a figure of the fit and forecast.
    :param periods: list of periods to fit seasonality e.g day, month, year is [1./1,1./30,1./365]
    :param upper_period: if no periods supplied, only fit seasonality up to 'upper_period'
        e.g 30 for max 1 month seasonality.

    :return: A 1D array of the predicted forecast (length = npoints + predict or len(dates_x)
    if manually specified
    '''


    npoints, ncomponents = np.shape(x_components)

    #normalise the x_components onto the same scale as the x time series
    mean_components, sd_components   = np.mean(x_components,axis=0), np.std(x_components,axis=0)
    mean_drive,sd_drive              = np.mean(x), np.std(x)


    times_all,dates_all = select_times(dates_x, predict=predict, datesmin_custom=None)
    for i in range(ncomponents+1):
        if i == 0:
            xpredict = x
            model = 'fourier'
        else:
            xpredict = x_components[:,i-1]
            model = 'interp'
        yg_tot, sg_tot, rmd, time_total = predict_seasonality(xpredict, dates=dates_x[:npoints],
                                                              dates_output=dates_all,
                                                              periods=periods, upper_period=upper_period, predict=0,
                                                              stats_return=True, iterations=1,
                                                              model=model,interpolate=True)
        if i == 0:
            y_main = yg_tot
            y_parts = np.zeros((len(yg_tot), ncomponents))
        else:
            y_parts[:,i-1] = yg_tot


    # if no components supplied, just stick with historic prediction
    if (ncomponents == 0):
        output_dictionary = {'predictions': y_main[-predict:],
                             'times predictions': time_total[-predict:],
                             'dates predictions': dates_all[-predict:],
                             'all': y_main,
                             'times all': time_total,
                             'dates all': dates_all}
        return (output_dictionary)

    #Search for and apply lags and calculate cross correlation coefficients.
    #can also force a lag if supplied in inpu lag list argument
    r2max = []
    lagsave = []
    for i in range(ncomponents):
        lcflow = np.array([time_total, y_main]).T
        lcarb = np.array([time_total, y_parts[:, i]]).T
        lag_ccf, ccf_save, lagpeak, ccfpeak, lagcent = ccf_frrss(lcflow, lcarb, resolution=1.0,
                                                                 fraction_rss=0.8, nsim=500,
                                                                 centroid_frac=0.8, flux_randomisation=0)

        if lag is None:
            id = np.argmax(lagpeak)
            r2max.append(ccfpeak[id])
            lagnow = - np.int(np.max(lagpeak))
        else:
            id     = np.argmin(np.abs(lag_ccf - - lag[i]))
            lagnow = lag[i]
            ccf_save = np.array(ccf_save)
            r2now = np.max(ccf_save[:, id])
            r2max.append(r2now)

            print('lag bug',lag[i],id,lag_ccf[id],np.shape(lag_ccf),np.shape(ccf_save),r2now)
            #print(lag_ccf)

        lagsave.append(lagnow)
        y_parts[:, i] = np.roll(y_parts[:, i], -lagnow)
    # first identify the importance of each signal to the main signal
    lead_max = np.abs(np.min(lagsave))
    r2max = np.array(r2max)
    lagsave = np.array(lagsave)
    idx_lead = np.where(lagsave < 0)[0]
    if len(idx_lead) == 0:
        print('no lead relationship identified: Aborting with historic forecast only')
        return(y_main)



    parms, cov, explained_variance, mse, importance = constituents_fit(y_main[lead_max:], y_parts[lead_max:, idx_lead],
                                                       trainfrac=0.8, verbose=False, diagnostic=False)



    # predict the combined forecast
    # combine with a r2 weighted average
    #r2wa considers only historioc flows if all constituent r^2 values --> 0
    #and considers increasingly the contributions from compoents for larger r^2 values
    #if components account for all explained variance, prediction does not use the historic data
    #(no need)

    y_predict_total = np.array(y_main)
    y_parts = (y_parts - mean_components)*sd_drive/sd_components + mean_drive

    print('mean drive', mean_drive)
    print('sd drive', sd_drive)
    print('mean components', mean_components)
    print('sd components', sd_components)
    print('lags...',lagsave)
    print('lead max',lead_max)
    print('r2max',r2max)
    for i in range(1,lead_max):
        #identify all components that lead flows by at least 'i' days
        id_include = np.arange(len(lagsave))#np.where(lagsave < -i)[0]
        top =   y_predict_total[-i]*(1.-explained_variance) + np.sum( r2max[id_include] * y_parts[-i,id_include])
        bottom = 1. - explained_variance + np.sum(r2max[id_include])

        y_predict_total[-i] = top / bottom


    if diagnostic == True:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(time_total, y_predict_total, label='model & predict')
        ax1.plot(np.arange(npoints),x,label='data')
        plt.legend()
        plt.show()


    output_dictionary = {'predictions':y_predict_total[-predict:],
                         'times predictions': time_total[-predict:],
                         'dates predictions': dates_all[-predict:],
                         'all':y_predict_total,
                         'times all': time_total,
                         'dates all': dates_all}
    return (output_dictionary)







































def figure_prioritize(prioritize_seasonality_output, title='./figures/figure_prioritize.pdf'):
    amplitudes = prioritize_seasonality_output['amplitudes']
    rmse = prioritize_seasonality_output['rmse']
    f_sort = prioritize_seasonality_output['frequencies']

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(amplitudes, label='amplitudes')
    ax1.set_xlabel('highest frequency harmonic (days)')
    ax1.set_ylabel('amplitude')

    ax1 = fig.add_subplot(312)
    ax1.plot(rmse, label='rmse (small numbers good)')
    ax1.set_xlabel('highest frequency harmonic (days)')
    ax1.set_ylabel('rmse')

    ax1 = fig.add_subplot(313)
    ax1.plot(f_sort,amplitudes, label='ampltiude')
    ax1.set_xlabel('highest frequency harmonic (days)')
    ax1.set_ylabel('amplitude')
    plt.tight_layout()
    plt.savefig(title)






def figure_optimize(optimize_seasonality_output, title='./figures/figure_optimize.pdf'):
    period = optimize_seasonality_output['periods']
    variance_components = optimize_seasonality_output['explained_variance']
    rmse = optimize_seasonality_output['rmse']
    bic = optimize_seasonality_output['BIC']

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(period, variance_components, label='explained variance (high numbers good)')
    ax1.set_xlabel('highest frequency harmonic (days)')
    ax1.set_ylabel('explained variance')

    ax1 = fig.add_subplot(312)
    ax1.plot(period, rmse, label='rmse (small numbers good)')
    ax1.set_xlabel('highest frequency harmonic (days)')
    ax1.set_ylabel('rmse')

    ax1 = fig.add_subplot(313)
    ax1.plot(period, bic, label='BIC (small numbers good)')
    ax1.set_ylabel('BIC')
    ax1.set_ylabel('highest frequency harmonic (days)')
    plt.tight_layout()
    plt.savefig(title)


def figure_seasonality(dates, y, dates_predict, y_predict, title='show'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(dates, y, label='data')
    ax1.plot(dates_predict, y_predict, label='model')
    ax1.set_xlabel('date')
    ax1.set_ylabel('time series')
    plt.legend()
    if title == 'show':
        plt.show()
    else:
        plt.savefig(title)





def figure_model(dates, values, labels=[],
                 xlabel='Date',figure_title='show'):
    """
    Produce a figure with flow data, historic and combined forecasts

    :param dates: list of array of time stamps
    :param values: list of array of values
    :param labels: list labelling each time series
    :param xlabel: The time axis label
    :param figure_title: 'show' to plot to screen or file name to save

    :return: Either plot to screen (figure_title = 'show' or save figure)
    """

    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA
    import matplotlib.pyplot as plt
    import matplotlib.pylab
    plt.clf()
    matplotlib.pylab.rcParams['xtick.major.pad'] = '12'
    n_components = len(dates)
    host = host_subplot(1,1,1, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    p1, = host.plot(dates[0], values[0], label=labels[0])
    host.axis["left"].label.set_color(p1.get_color())
    host.set_ylabel(labels[0])
    host.annotate(xlabel, (0.5, -0.2), xycoords='axes fraction',
                 horizontalalignment='right')

    #host.set_xlabel(xlabel,labelpad=20)

    print('labels..',labels)
    for i in range(1,n_components):
        par = host.twinx()
        offset = 60*(i-1)
        if i == 1:
            new_fixed_axis = par.get_grid_helper().new_fixed_axis
        par.axis["right"] = new_fixed_axis(loc="right", axes=par,
                                        offset=(offset, 0))
        par.axis["right"].toggle(all=True)
        p1, = par.plot(dates[i],values[i],label=labels[i])
        par.axis["right"].label.set_color(p1.get_color())
        par.set_ylabel(labels[i])

        print(i,labels[i])

    plt.setp(host.axis['bottom'].major_ticklabels, rotation=45)
    host.legend()
    plt.draw()
    plt.style.use('dark_background')


    plt.tight_layout()
    if figure_title == 'show':
        plt.show()
    else:
        plt.savefig(figure_title)
    plt.clf()
    return()




# make sure data is on same time scale
# looks and fits trends on timescales defined in periods. Predicts 'predict' time steps off the end of both data
# upper period states the maximum upper frequency if periods not supplied
def remove_seasonality(arbs, flows_df, dates=None, periods=None, upper_period=60., predict=10, fig_title=None):
    y = [arbs, flows_df]

    print('remove seasonality function')

    y_season = []
    y_season_gone = []
    y_season_predict = []
    y_season_total = []
    uncertanties_total = []
    r2 = []
    idx = 0
    dates_total = []

    for ynow in y:
        ny = len(ynow)
        if dates is not None:
            time, dates_total = select_times(dates, predict=predict)
        else:
            time = np.arange(ny)

        time_total = np.arange(time[0], time[-1] + 1 + predict, 1)

        yg_tot, sg_tot, rmd = \
            predict_seasonality(ynow, dates=dates, periods=periods, upper_period=upper_period,
                                predict=predict, stats_return=True)
        y_interpolate = np.interp(time, time_total, yg_tot)
        y_season.append(yg_tot[:ny])
        y_season_predict.append(yg_tot[ny:])
        y_season_total.append(yg_tot)
        uncertanties_total.append(sg_tot)
        y_season_gone.append(ynow - yg_tot[:ny])
        r2.append(r2_score(yg_tot[:ny], ynow))
        idx = idx + 1
        if fig_title is not None:
            figure_seasonality(time, ynow, time_total, yg_tot,
                               title=fig_title[:-4] + '_' + np.str(idx) + fig_title[-4:])
    return y_season, y_season_gone, y_season_predict, r2, dates_total, y_season_total, y_interpolate, uncertanties_total


def print_metrics(arbs, flows_df):
    arbs, flows_df = normalise(arbs, flows_df)

    ccf_score = ss.correlate(arbs, flows_df) / len(flows_df)
    # ccf < 0 => means x leads y
    print('lag: ', np.argmax(ccf_score) - (len(ccf_score) / 2))
    print('ccf: ', np.max(np.nan_to_num(ccf_score)))



def normalise(arbs, flows_df):
    arbs = arbs.fillna(arbs.mean())
    flows_df.fillna(flows_df.mean())

    arbs = (arbs - arbs.mean()) / arbs.std()
    flows_df = (flows_df - flows_df.mean()) / flows_df.std()

    return arbs, flows_df


def load_ob_data(months_ahead):
    ob_arb = pd.read_csv(path + 'gasoline_consolidated.csv')
    ob_arb = ob_arb[(ob_arb['from_region'] == 'NWE') & (ob_arb['to_region'] == 'USAC/ECCAN')]
    a = ob_arb[(ob_arb['from_region'] == 'NWE') & (ob_arb['to_region'] == 'USAC/ECCAN')]
    arb_string = 'month' + str(months_ahead) + '_arb'
    ob_arb = pd.DataFrame([a['date'], a[arb_string]]).T
    ob_arb['date'] = pd.to_datetime(ob_arb['date'])
    ob_arb['ob_arb'] = ob_arb[arb_string]
    print('Loaded ',len(ob_arb),' records')
    return ob_arb[['date', 'ob_arb']]






def load_all_data(months_ahead=0, flows_offset=60):
    arb = pd.read_csv(path + 'calculated_arbs.csv')
    arb.rename(columns={'arb': 'vortexa_arb'}, inplace=True)
    arb['date'] = pd.to_datetime(arb['date'])
    arb = arb[arb['months_ahead'] == months_ahead]

    flows = pd.read_csv(path + 'vortexa_gasoline_naphtha_1day/gasoline_nwe_usaccec.csv')[
        ['Quantity loaded on', 'quantity (bbl)']]
    flows['flows_tonnage'] = flows['quantity (bbl)'] / GASOLINE_BARRELS_PER_MT
    flows['load_date'] = pd.to_datetime(flows['Quantity loaded on']) - pd.offsets.Day(flows_offset)

    arb_flows = arb.merge(flows, left_on='date', right_on='load_date')
    arb_flows = arb_flows.sort_values(by='date')
    # load the OB arb data
    ob_arb = load_ob_data(months_ahead)
    ob_arb_flows = arb_flows.merge(ob_arb, left_on='date', right_on='date')

    flows_not2_usaccec = pd.read_csv(path + 'vortexa_gasoline_naphtha_1day/gasoline_nwe_NOT2_usaccec.csv')[
        ['Quantity loaded on', 'quantity (t)']]
    flows_not2_usaccec.rename(columns={'quantity (t)':'tonnes not_2_usaccec'},inplace=True)
    flows_not2_usaccec['Quantity loaded on'] = \
        pd.to_datetime(flows_not2_usaccec['Quantity loaded on']) - pd.offsets.Day(flows_offset)
    ob_arb_flows = ob_arb_flows.merge(flows_not2_usaccec, left_on='date',right_on='Quantity loaded on')
    return ob_arb_flows.sort_values(by='date')


def arb_smoothing_function(series, window):
    return series.rolling(window).mean()
    # return 0.5 * (series.rolling(window).mean() + series.rolling(window).max())
    # return series.ewm(span=window).mean()


def figure_model_time_series(dates_data, y_data, dates_model, y_model, train_dates_model, train_y_model,
                             xlabel='date', ylabel='flow (MT)', title='Flows time series',
                             fig_tit='fig_flows_timeseries.pdf',
                             extra_line=[], ann_extra_line=[],
                             label=['historic flows', 'model flows', 'trained model flows']):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(dates_model, y_model, label=label[1], color='k')
    ax1.plot(dates_data, y_data, label=label[0], color='y', ls='--')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(fig_tit)










def general_timeseries_figure(dates,values,
                              xlabels = 'Date',
                              ylabels = 'Flows (MT)',
                              figure_title='show',labels=None,
                              v_lines = None):
    """
    Plot multiple time series on a single axis

    :param dates: list of timestamps for each time series
    :param values: list of numpy 1D arrays (1 for each time series)
    :param labels: labels for each time series
    :param figure_title show to screen if 'show', else plot to figure
    :param v_lines: add extra vertical lines to the plot

    :return: wither show plot or save to file (figure_title)
    """
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabels)
    ax1.set_ylabel(ylabels)
    for i in range(len(dates)):
        if labels is None:
            lab = ''
        else:
            lab = labels[i]
        ax1.plot(dates[i],values[i],label=lab)
    ax1.tick_params(axis='x', rotation=45)
    if v_lines is not None:
        ylims = list(ax1.get_ylim())
        [ax1.plot(v_lines[idx],ylims,label=None,ls='--',color='white') for idx in range(len(v_lines))]
        ax1.set_ylim(ylims)

    plt.legend(fontsize='x-small')
    plt.tight_layout()
    if figure_title == 'show':
        plt.show()
    else:
        plt.savefig(figure_title)
    return()











def figure_derivative_model_time_series(dates_data, y_data, dates_model, y_model, train_dates_model, train_y_model,
                                        xlabel='date', ylabel='flow change (MT) relative to previous month',
                                        title='month to month change in flows',
                                        fig_tit='fig_flows_timeseries.pdf',
                                        extra_line=[], ann_extra_line=[],
                                        label=['historic flows', 'model flows', 'trained model flows']):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    def dates_to_array(dates, dates_min_in=False):
        if dates_min_in == False:
            dates_min = min(dates)
        else:
            dates_min = dates_min_in
        timedelta_dates = pd.TimedeltaIndex(dates - dates_min)
        days_output = np.array(timedelta_dates.days, dtype='float')
        return days_output

    diff_dates_data = dates_data[1:]
    dates_min = min(min(dates_data), min(dates_model))
    time_data = dates_to_array(dates_data, dates_min_in=dates_min)
    time_model = dates_to_array(dates_model, dates_min_in=dates_min)
    diff_data = y_data[1:] - y_data[:-1]
    model_interp = np.interp(time_data, time_model, y_model)
    diff_model_interp = model_interp[1:] - model_interp[:-1]
    sig_y = np.abs(diff_model_interp - diff_data)
    ax1.plot(diff_dates_data, diff_data, label=label[0], color='y', ls='--')
    ax1.plot(diff_dates_data, diff_model_interp, label=label[1], color='k')


    upup = np.where((diff_data > 0) & (diff_model_interp > 0))[0]
    nupup = np.shape(upup)[0]

    downdown = np.where((diff_data < 0) & (diff_model_interp < 0))[0]
    ndowndown = np.shape(downdown)[0]

    npoints = np.shape(diff_data)[0]

    print('up up', nupup)
    print('down down', ndowndown)
    print('agreement = ', 100 * (np.float(nupup) + ndowndown) / npoints)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)

    ax1.tick_params(axis='x', rotation=45)

    ax1.legend()
    plt.tight_layout()
    plt.savefig(fig_tit)


def apply_lag(x, y, lag):
    """
    Roll y by lag (-ve indicates y leads) and return overlapping points

    :param x: Input x values (numpy 1D array)
    :param y: Input y values (numpy 1D array)
    :param lag: Integer number of time steps -ve indicates y leads x
    :return: x_common,y_common,correlation_coefficient
    """
    # shift y by the lag and select only overlapping points
    nx = len(x)
    if lag < 0:
        idx_common = np.arange(-lag, nx, 1)
    else:
        idx_common = np.arange(0, nx - lag)
    y_common = np.roll(y, -lag)[idx_common]
    x_common = x[idx_common]
    correlation_coefficient = np.corrcoef(x_common, y_common)[0, 1]
    return (x_common,y_common,correlation_coefficient)



def correlation_figure(x,y,lag,
                       order = 1,xylabs=('x','y'),figure_title='show',
                       bin = True,
                       global_title='correlation analysis'):
    """
    Compute the cooreltion between x and y (with a given lag) and plot the result.

    :param x: Input x values (numpy 1D array)
    :param y: Input y values (numpy 1D array)
    :param lag: Integer number of time steps -ve indicates y leads x
    :param order: polynomial order to fit to data (default 1)
    :param xylabs: labels for x and y axis (tuple(xlab,ylab))
    :param figure_title: the output figure title (if 'show' then plot to screen)
    :param bin: If true, bin the x,y data into 10 x bins before fitting model.
        The displayed points will be unaltered but the binned points will be used to calculate
        the uncertainty. This is all because linear regression uncertainty envelopes
        screw up when you have lots of points sampled around the same x value as here.
        Default error envelopes too small. Rebinning goes at least part of the way to fixing this.
    :return Either plot figure to screen (if figure_title = 'show') or save to file:
    """
    #identify overlapping elements and shift by lag
    x_common, y_common, correlation_coefficient = apply_lag(x,y,lag)

    #make figure
    xmin,xmax,xsd = np.min(x_common),np.max(x_common),np.std(x_common)
    xgrid = np.linspace(xmin-xsd,xmax+xsd,len(x_common)*10)

    #produced binned results for reliable error envelopes
    if bin is True:
        from scipy.stats import binned_statistic as bs
        y_common_fit, bin_edges, bin_number  = bs(x_common,values=y_common,statistic='mean')
        y_common_sig, bin_edges, bin_number = bs(x_common, values=y_common, statistic=np.std)
        x_common_fit = (bin_edges[1:]+bin_edges[:-1])/2
    else:
        y_common_fit = y_common
        x_common_fit = x_common
        y_common_sig = None

    yg_med, yg_lo, yg_hi, cov, cisq, cisq_red, bic, aic, rmd, r2 =\
        fit(x_common_fit, y_common_fit, sig=y_common_sig, order=order,
        xgrid=xgrid, confidence=0.3173, nits=20000, figure_title='', verbose=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x_common,y_common,label=None,color='b')
    ax1.plot(xgrid,yg_med,label=None,color='cyan')
    ax1.fill_between(xgrid,yg_lo,yg_hi,alpha=0.4,color='cyan')
    ax1.set_xlabel(xylabs[0])
    ax1.set_ylabel(xylabs[1])
    ax1.annotate(r'$r_c = '+np.str(np.round(correlation_coefficient,2))+'$',
                  (0.95, 0.95), xycoords='axes fraction',
                 horizontalalignment='right', color='cyan')
    ax1.set_title(global_title)
    plt.style.use('dark_background')

    if figure_title == 'show':
        plt.show()
    elif figure_title == '':
        pass
    else:
        plt.savefig(figure_title)
    return()





def figure_flow_arbs(x, y, fitx, fity, xlabel='flow MT', ylabel='arb (dollars / MT)', title='Flows Vs Arbs regression',
                     fig_tit='fig_arbs_flows_regression.pdf'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(fitx, fity)
    ax1.plot(x, y, marker='o',ls=None)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    fity_itp = np.interp(x, fitx, fity)
    r2 = r2_score(y, fity_itp)
    ax1.annotate(r'$r^2 = ' + np.str(np.round(r2, 2)) + '$', (0.5, -0.3), xycoords='axes fraction',
                 horizontalalignment='right', color='b')
    ax1.set_title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    if fig_tit is None:
        pass
    elif fig_tit == 'show':
        plt.show()
    else:
        plt.savefig(fig_tit)


def figure_flow_arbs_overstate(x, y, fitx, fity, dates, xlabel='flow (MT)', ylabel='arb (dollars / MT)',
                               title='Flows Vs Arbs regression', fig_tit='fig_arbs_flows_overstate.pdf'):
    fitarb_itp = np.interp(y, fity, fitx)
    arb_overstate = x - fitarb_itp
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(dates, arb_overstate)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_tit)






class forecast:
    """
    groups the functions above into a single class with
    fit, predict, plot arguments to aid usability.

    """


    dates         = []
    x             = []
    x_components  = []
    periods       = [730,365,180,90]
    days_forecast = 90.0
    model_output  = []
    dates_output  = []
    model_predictions = []
    dates_predictions = []
    times_output      = []


    #input dates
    def set_dates(self,dates):
        self.dates =  dates

    #input time series
    def set_main_timeseries(self,x):
        self.x = x

    #input additional contributing time series
    def set_contributing_timeseries(self,x_components):
        self.x_components = x_components

    #fit the mode to the historical + additional components
    def fit(self):
        output = combine_signal_lag(self.x, self.x_components, self.dates,
                           lag=None, predict=0,diagnostic = False,
                       periods = self.periods, upper_period = 30)
        self.model_output      = output['all']
        self.dates_output      = output['dates all']
        self.times_output      = output['times all']
        self.dates_predictions = output['dates predictions']
        self.model_predictions = output['predictions']

    #predict 'days' number of days into the future
    def predict(self,days = 90):
        output = combine_signal_lag(self.x, self.x_components, self.dates,
                           lag=None, predict=days,diagnostic = False,
                       periods = self.periods, upper_period = 30)
        self.model_output      = output['all']
        self.dates_output      = output['dates all']
        self.times_output      = output['times all']
        self.dates_predictions = output['dates predictions']
        self.model_predictions = output['predictions']
        return(output['predictions'])



    #produce correlation figure
    def plot(self):
        plt.close('all')
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.dates, self.x,marker='o',ls=None,label='historic')
        ax1.plot(self.dates_output, self.model_output, label='prediction')
        ax1.set_xlabel('date')
        ax1.set_ylabel('flow')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()


    def get_dates(self):
        return(self.dates)

    def get_model_dates(self):
        return(self.dates_output)

    def get_model_values(self):
        return(self.model_output)

    def get_main_timeseries(self):
        return(self.x)

    def get_contributing_timeseries(self):
        return(self.x_components)

    def get_periods(self):
        return(self.periods)

    def get_predictions(self):
        return(self.model_predictions)

    def get_prediction_dates(self):
        return(self.dates_predictions)

    def get_dates_as_times(self):
        return(self.times_output)



#test the forecast class by generating fake flow and fkae arb that leads by 3 days
# add on separate 'noise' processes to each process to simulate uncertainty
if __name__ == '__main__':

    #generate fake data
    n = 3 * 365
    dt = 1
    time = np.arange(0, n, dt)
    predict = 100
    time_extra = np.arange(0, time[-1] + (predict + 1) * dt, dt)
    noise = 0.3
    y_flows_true = np.sin(2 * np.pi * 1. / 365 * time_extra) + 0.3 * np.sin(2 * np.pi * 1. / 90 * time_extra)
    y = y_flows_true + noise * np.random.randn(n + predict)
    y_train = y[:-predict]
    # select an arb that leads by 3 days
    y_arb = np.roll(y_flows_true, -100) + noise * np.random.randn(n + predict)
    y_components = np.zeros((n + predict, 1))
    y_components[:, 0] = y_arb
    y_components_train = y_components[:-predict]
    # specify the dates of the observations (list of dataframes)
    today = pd.datetime.now()
    dates = pd.date_range(start=today, periods=n)

    #this is how to use the forecast class
    a = forecast()
    a.set_dates(dates)
    a.set_main_timeseries(y_train)
    a.set_contributing_timeseries(y_components_train)
    a.predict(days=90)
    a.plot()
    plt.show()