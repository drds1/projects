import numpy as np
import pandas as pd


class timeseries_prep:

    def __init__(self):
        self.forecast = 0
        self.timeseries = None
        self.padd = 'mean'
        self.name_main = None
        self.covariate_info = None
        self.timeseries = pd.DataFrame({})

    def add_component(self,component,name,
                      lag = 0,
                      ismain = False):
        '''
        adding all components should use this code
        :param component:
        :param name:
        :return:
        '''
        self.timeseries[name]=component
        if self.covariate_info is None:
            self.covariate_info = {'name':[],
                                    'lag':[],
                                    'main':[]}

        self.covariate_info['name'].append(name)
        self.covariate_info['lag'].append(lag)
        self.covariate_info['main'].append(ismain)


    def add_main(self,component,name='main'):
        '''
        add the main component to be fitted this will take the position of the first column
        :param component:
        :param name:
        :return:
        '''
        self.name_main = name

        vals = np.append(component, np.nan * np.ones(self.forecast))
        self.add_component(vals,name,
                           lag=0,
                           ismain=True)
        self.nhistoric = len(component)
        self.ntot = self.nhistoric + self.forecast


    def add_covariate(self,component,name, lags = np.arange(10)):
        '''
        add a component to the time series analysis padd up to forecast with the mean or median of timeseries
        for forecasting
        :param component:
        :param name:
        :return:
        '''
        if self.name_main is None:
            raise Exception('must add the main fit y-axis before adding covariates (self.add_main)')
        vals = np.array(component,dtype=float)
        if self.padd == 'mean':
            x = np.mean(vals)
        elif self.padd == 'median':
            x = np.median(vals)
        vals = np.append(vals,x*np.ones(self.forecast))

        '''
        add autoregression components
        '''
        for l in lags:
            self.add_component(np.roll(vals,l), np.str(l) + ' step ' + name,
                          lag=l,
                          ismain=False)

    def add_trend(self,order=0):
        '''
        add a trend component to the fit
        MUST FIRST ADD ANY COVARIATES ABOVE
        :return:
        '''
        i = np.arange(self.ntot)
        vals = ((2.0*i)/self.ntot - 1.0)**order
        self.add_component(vals,name='trend order '+np.str(order),
                           ismain=False,lag = np.inf)


    def add_seasonality(self,period):
        '''
        add seasonal sin and cosin components to fit
        :param period:
        :return:
        '''
        i = np.arange(self.ntot)
        x = 2*np.pi*i/period
        sin = np.sin(x)
        cos = np.cos(x)
        self.add_component(sin,name='sin '+np.str(period)+' step period',lag = np.inf)
        self.add_component(cos, name='cos ' + np.str(period) + ' step period',lag = np.inf)


    def training_cross_validation(self):
        '''
        identify which points can be used and which must be excluded for cross validation
        e.g if doing a 2 step forecast, can only use points of 2-step lag or higher or the
        seasonal and trend components. remove_point tells you which covariates to omit
        with sliding corss validation window for each time series point
        :return: 2d array of allowed covariates.
        '''
        nx = np.shape(self.timeseries.values[0,:])[0]
        self.training_remove_points = np.ones((nx,self.nhistoric))
        lags = np.array(self.covariate_info['lag'].values[:],dtype=float)
        lags[lags==np.inf] = 0
        for idx_omit in range(self.nhistoric):
            self.training_remove_points[:,idx_omit] = lags + idx_omit

    def final_cross_validation(self,start_point,forecast_step = 10):
        '''
        which points to omit for the final cross validation this time remove one point off the end
        not all points allowed e.g can only use 2step lag or greater for 2-step forecast
        other covairates set to mean
        :return:
        '''
        last_point = np.arange(start_point,self.nhistoric) - forecast_step
        lags = np.array(self.covariate_info['lags'],dtype=float)
        idx_notallowed = np.where(lags < forecast_step)
        for l in last_point:
             forecast_point = l + forecast_step
             covariates = self.timeseries.values[forecast_point,1:]
             if self.padd == 'mean':
                 x = np.mean(self.timeseries.values[:l,idx_notallowed],axis=0)
             elif self.padd == 'median':
                 x = np.median(self.timeseries.values[:l, idx_notallowed], axis=0)
             covariates[idx_notallowed-1] = x






if __name__ == '__main__':
    '''
    test rfr on some data
    '''
    import numpy as np
    import sys
    sys.path.append("/Users/david/github_datascience/projects/fake_data")
    sys.path.append("/Users/david/github_datascience/projects/regressor_comparison")
    from fake_data import *
    from regressors import *


    #generate fake data
    a = fake_data()
    a.covariates = 4
    a.npoints = 200
    seasonal_periods = [52.,12.,16.,26.]
    trend_order = 4
    a.initialise()
    a.add_covariates(importances=[1., 0.0, 0.0, 0.0], iseed=343435)

    #now do timeseries prep
    p = timeseries_prep()
    p.add_main(a.driver, name='main')
    ncov = np.shape(a.extra_covariates[0,:])[0]
    for i in range(ncov):
        p.add_covariate(a.extra_covariates[:,i], 'component '+np.str(i+1), lags=np.arange(10))
    for i in range(trend_order):
        p.add_trend(order=i)
    for period in seasonal_periods:
        p.add_seasonality(period=period)

    x = rfr()
    #set required parameters
    x.feature_list = list(p.timeseries.columns)[1:]
    x.covariates = p.timeseries.values[:,1:]
    x.ymain = p.timeseries.values[:,0]

    #perform fit and make plots
    x.split_train_test()
    x.test_all_methods()


