import numpy as np
import pandas as pd
import linvpy.linvpy as lp
import scipy
import matplotlib.pylab as plt
import prediction_functions.signal_decomp as dcomp
from prediction_functions.evaluate_model import *
from prediction_functions.correlation_figure import *
from prediction_functions.apply_lag import *

#compute the hat matrix for a regularise GLM problem
def hat(Xin,regularize_weight = 0):
    X    = np.matrix(Xin).T
    XT   = X.T
    w = np.matrix( regularize_weight*np.identity(np.shape(X)[1]) )
    XTX_i = np.linalg.inv(XT*X + w)
    b = X*(XTX_i*XT)
    return(b)



#covariance plot
def fn_plot_covariance(fig,ax,cov,xlabel='lag (days)',xticks=None):
    std = np.sqrt(np.diag(cov))
    nparm = np.shape(cov)[0]
    cor = np.zeros((nparm,nparm))
    for i in range(nparm):
        for j in range(nparm):
            cor[i,j] = cov[i,j]/std[i]/std[j]
    ishow = ax.imshow(cor,aspect='auto')
    ax.set_xlabel(xlabel)
    if xticks is not None:
        labels = ['']*len(xticks)
        for i in range(0,len(xticks),20):
            labels[i] = np.str(xticks[i])
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(len(xticks)))
        ax.set_yticklabels(labels)
    cbar = fig.colorbar(ishow,cmap='RdBu')
    cbar.set_label('Correlation')

def fn_plot_response(ax,lag_grid,response_grid):
    ax.plot(lag_grid,response_grid)
    ax.set_xlabel('lag (days)')
    

def make_A(timeseries_ar,timeseries_nonar,lagrange,nepoch):
    n_ar = len(timeseries_ar)
    n_nonar = len(timeseries_nonar)
    idlo, idhi = lagrange
    idlag = np.arange(idlo, idhi + 1, 1)
    lenlag = len(idlag)
    A_out = np.zeros((0,nepoch))
    for i in range(n_ar):
        ts = timeseries_ar[i]
        A = np.tile(ts, lenlag)
        A = np.reshape(A, (lenlag, nepoch))
        for i2 in range(lenlag):
            lagnow = idlag[i2]
            x = np.roll(A[i2, :], idlag[i2])
            if lagnow > 0:
                x[:lagnow] = A[i2,0]
            elif lagnow < 0:
                x[-lagnow:] = A[i2,-1]
            A[i2, :] = x
        A_out = np.vstack((A_out,A))
    idx_nonar = np.arange(np.shape(A_out)[0],np.shape(A_out)[0]+n_nonar)
    for i in range(n_nonar):
        ts = timeseries_nonar[i]
        A  = np.zeros((1,nepoch))
        A[0,:] = ts
        A_out = np.vstack((A_out,A))

    #ncomponents, ndata = np.shape(A_out)
    #A_means = np.mean(A_out, axis=1)
    #A_sd    = np.std(A_out,axis = 1)
    #A_out = A_out - np.reshape(np.tile(A_means,ndata),(ncomponents,ndata))
    #A_out = A_out/np.reshape(np.tile(A_sd,ndata),(ncomponents,ndata))
#
    #id_zero = np.where(A_sd == 0)[0]
    #A_out[id_zero,:] = 1


    return(A_out)


#recast driving time series into N X Nlag matrix A for linvpy process
class RLI:

    def __init__(self):
        #input parameters
        self.response_timeseries = 0
        self.lagrange = [-1,1]#[-20,20]
        self.response = 0
        self.result = 0
        self.A = None
        self.regularize = 'minimize squares'
        self.regularize_weight = 0
        self.idx_response = []
        self.confidence_limits = [16,50,84]
        self.idx_non_ar = []
        self.idx_ar = []
        self.labels_ar = []
        self.labels_non_ar = []
        self.predictions = np.array([])
        self.input_ar_timeseries = []
        self.input_nonar_timeseries = []
        self.ncomponents = 0
        self.n_nonar = 0
        self.error_envelopes = None
        self.idx_nonar_count_fourier = []
        self.idx_non_ar_count = []
        self.idx_nonar_fourier = []
        self.idx_nonar_trend = []
        self.n_fourier = 0
        self.periods = None
        self.n_ar = 0
        self.dates_input_nonar_timeseries = []
        self.dates_input_ar_timeseries = []
        self.dates_global = None
        self.labels_fourier = []
        self.npredict = 0
        self.fourier_component_results = None
        self.n_trend = 0
        self.labels_trend = []
        self.idx_nonar_count_trend = []
        self.trend_component_results = None
        self.trend_component_residuals = None
        self.trend_explained_variances = None
        self.ar_response = []
        self.predictions = None
        self.dates_prediction = None
        self.mean_ar_timeseries = []
        self.mean_nonar_timeseries = []
        self.mean_response_timeseries = None
        self.sd_response_timeseries = None
        self.sd_ar_timeseries = []
        self.sd_nonar_timeseries = []

    def add_component(self,timeseries,name=None,kind='normal', dates = None):
        ymean,ysd = np.mean(timeseries), np.std(timeseries)
        #timeseries = (timeseries - ymean)/ysd
        self.mean_nonar_timeseries.append(ymean)
        self.sd_nonar_timeseries.append(ysd)
        self.input_nonar_timeseries.append(timeseries)
        self.ndata = len(timeseries)
        if self.dates_global is None:
            self.dates_global = np.arange(self.ndata)
        if dates is not None:
            self.dates_input_nonar_timeseries.append(dates)
            self.dates_global = dates
        if name is None:
            lab = 'Non AR \ncomponent '+np.str(len(self.idx_non_ar))
        else:
            lab = name
        self.ncomponents = self.ncomponents + 1
        if kind is 'fourier':
            self.idx_nonar_count_fourier.append(self.n_nonar)
            self.n_fourier = self.n_fourier + 1
            self.labels_fourier.append(lab)
        elif kind is 'trend':
            #self.idx_nonar_count_trend.append(self.n_trend)
            self.n_trend = self.n_trend + 1
            self.labels_trend.append(lab)
            self.idx_nonar_count_trend.append(self.n_nonar)
        else:
            self.idx_non_ar_count.append(self.n_nonar)
            self.labels_non_ar.append(lab)
        self.n_nonar = self.n_nonar + 1


    def add_ar_component(self,timeseries, name = None, dates = None):
        ymean,ysd = np.mean(timeseries), np.std(timeseries)
        timeseries = (timeseries - ymean)/ysd
        self.mean_ar_timeseries.append(ymean)
        self.sd_ar_timeseries.append(ysd)
        self.input_ar_timeseries.append(timeseries)
        self.ndata = len(timeseries)
        if self.dates_global is None:
            self.dates_global = np.arange(self.ndata)
        if dates is not None:
            self.dates_input_ar_timeseries.append(dates)
            self.dates_global = dates
        idlag = np.arange(self.lagrange[0],self.lagrange[1]+1,1)
        self.lag_grid = idlag
        if name is None:
            self.labels_ar.append('AR \ncomponent '+np.str(len(self.idx_ar)))
        else:
            self.labels_ar.append(name)
        self.ncomponents = self.ncomponents + len(idlag)
        #for the lag calculation,
        #remember the weight indicees belonging to each auto regressive time series
        idhi = self.ncomponents
        idlo = idhi - len(idlag)
        self.idx_response.append([idlo,idhi])
        self.n_ar = self.n_ar + len(idlag)



    def interpolate(self,datesmain):
        t, dates_global = select_times(datesmain, predict=0, datesmin_custom=None)
        times_global = np.arange(t[0],t[-1]+1,1)
        self.ndata = len(times_global)
        self.dates_global = dates_global
        for i in range(self.n_ar):
            tnow,dnow = select_times(self.dates_input_ar_timeseries, predict=0, datesmin_custom=None)
            xnow = self.input_ar_timeseries[i]
            xnew = np.interp(times_global,tnow,xnow)
            self.input_ar_timeseries[i] = xnew
        for i in range(self.n_nonar):
            tnow,dnow = select_times(self.dates_input_nonar_timeseries, predict=0, datesmin_custom=None)
            xnow = self.input_nonar_timeseries[i]
            xnew = np.interp(times_global,tnow,xnow)
            self.input_nonar_timeseries[i] = xnew



    def add_trend(self,order = 1):
        self.ndata = np.shape(self.response_timeseries)[0]
        t = np.arange(self.ndata)
        for i in range(1,order+1):
            y = t**i
            self.add_component(y, name = 'trend order '+np.str(i),kind='trend')

    def add_fourier_components(self,periods, dates = None):
        self.ndata = np.shape(self.response_timeseries)[0]
        t = np.arange(self.ndata)
        self.periods = periods
        for p in periods:
            xs = np.sin(2*np.pi/p * t)
            xc = np.cos(2*np.pi/p * t)
            self.add_component(xs,name='sin period '+np.str(p),kind = 'fourier', dates = dates)
            self.add_component(xc, name='cos period ' + np.str(p), kind='fourier', dates = dates)

    def fit(self,A_recalc=True):
        self.mean_response_timeseries = np.mean(self.response_timeseries)
        self.sd_response_timeseries   = np.std(self.response_timeseries)
        self.response_timeseries = (self.response_timeseries - self.mean_response_timeseries)/\
                                   self.sd_response_timeseries


        if A_recalc is True:
            self.A = \
                make_A(self.input_ar_timeseries, self.input_nonar_timeseries, self.lagrange, self.ndata)
        self.idx_non_ar = [self.ncomponents - self.n_nonar + i for i in range(self.n_nonar)]
        self.ncomponents,self.ndata = np.shape(self.A)

        #perform fit
        parms, cov, r2, mse, importance = \
            dcomp.constituents_fit(self.response_timeseries, self.A.T,
                                   regularize=self.regularize,
                                   regularize_weight=self.regularize_weight)
        self.response = parms
        self.cov = cov
        self.result = np.dot(self.response,self.A) + self.mean_response_timeseries


        #fit statistics
        hat_matrix = hat(self.A, regularize_weight = self.regularize_weight)
        degrees_of_freedom = np.trace(hat_matrix)
        self.degrees_of_freedom = degrees_of_freedom
        self.hat_matrix = hat_matrix
        self.residuals = self.result - self.response_timeseries
        self.sum_of_square_residuals = np.sum(self.residuals**2)
        self.variance_response_timeseries = np.var(self.response_timeseries)
        self.explained_variance = 1. - np.var(self.residuals)/self.variance_response_timeseries
        self.AIC = self.sum_of_square_residuals + 2*self.degrees_of_freedom
        self.BIC = self.sum_of_square_residuals + \
                   2 * self.degrees_of_freedom * np.log( len(self.response_timeseries) )
        self.component_results = self.response*self.A.T

        self.component_residuals =  self.component_results - \
                                    np.reshape(np.tile(self.response_timeseries,self.ncomponents),
                                    (self.ncomponents,self.ndata)).T
        self.component_explained_variances = 1. - \
                                   np.var(self.component_residuals,axis = 0)/self.variance_response_timeseries

        #for the auto regressive components
        self.ar_component_results   = []
        self.ar_component_residuals = []
        self.ar_explained_variances = []
        for i in range(len(self.input_ar_timeseries)):
            sl = slice(self.idx_response[i][0],self.idx_response[i][1])
            self.ar_component_results.append( np.dot(self.response[sl],self.A[sl,:]) + self.mean_response_timeseries )
            self.ar_component_residuals.append( self.ar_component_results[i] - self.response_timeseries )
            self.ar_explained_variances.append( 1. -
                                               np.var(self.ar_component_residuals[i])/
                                               self.variance_response_timeseries )

        #for the fourier components
        if self.n_fourier > 0:
            a1 = np.array(self.idx_non_ar)
            a2 = self.idx_nonar_count_fourier
            sl = np.intersect1d(a1 - a1[0], a2) + a1[0]
            self.idx_nonar_fourier = sl
            self.fourier_component_results = np.dot(self.response[sl],self.A[sl,:]) + self.mean_response_timeseries
            self.fourier_component_residuals = self.fourier_component_results - self.response_timeseries
            self.fourier_explained_variances = ( 1. -
                                                   np.var(self.fourier_component_residuals)/
                                                   self.variance_response_timeseries )


        #for the other components
        self.nonar_explained_variances = []
        self.nonar_component_results = []
        self.nonar_component_residuals = []
        for i in range(len(self.idx_non_ar)):
            self.nonar_component_results.append( self.component_results[:,self.idx_non_ar[i]] + self.mean_response_timeseries)
            self.nonar_component_residuals.append( self.component_residuals[:,self.idx_non_ar[i]] )
            self.nonar_explained_variances.append( self.component_explained_variances[self.idx_non_ar[i]] )

        #for trend components
        if self.n_trend > 0:
            a1 = np.array(self.idx_non_ar)
            a2 = self.idx_nonar_count_trend
            sl = np.intersect1d(a1 - a1[0], a2) + a1[0]
            self.idx_nonar_trend = sl
            self.trend_component_results   = np.dot(self.response[sl],self.A[sl,:]) + self.mean_response_timeseries
            self.trend_component_residuals = self.trend_component_results - self.response_timeseries
            self.trend_explained_variances = ( 1. -
    np.var(self.trend_component_residuals)/
    self.variance_response_timeseries )


        if len(self.idx_non_ar) > 0:
            a1 = np.array(self.idx_non_ar)
            a2 = self.idx_non_ar_count
            sl = np.intersect1d(a1 - a1[0], a2) + a1[0]
            self.idx_non_ar = sl
            idtot = np.append(np.append(self.idx_non_ar,self.idx_nonar_fourier),self.idx_nonar_trend)
            x = np.arange(self.ncomponents)
            self.idx_ar = np.setdiff1d(x,idtot)
        else:
            self.idx_ar = np.arange(self.ncomponents)


    def get_ar_response(self):
        self.ar_response = []
        for lag_limits in self.idx_response:
            self.ar_response.append( self.response[lag_limits[0]:lag_limits[1]] )
            print('lag limits ',lag_limits)
        return(self.ar_response)

    def get_lags(self):
        #calculate lags for each autoregressive component
        self.lag_confidences = []
        self.ar_response = self.get_ar_response()
        for response in self.ar_response:
            response[response < 0] = 0
            csum = np.cumsum(response)
            lag = []
            for confidence in self.confidence_limits:
                lnow = np.where(csum/csum[-1] > confidence/100)[0]
                if len(lnow) == 0 and confidence == self.confidence_limits[0]:
                    lag.append( self.lag_grid[0] )
                elif len(lnow) == 0 and confidence == self.confidence_limits[-1]:
                    lag.append( self.lag_grid[len(self.lagrange)] )
                else:
                    lag.append( self.lag_grid[lnow[0]] )
            self.lag_confidences.append(lag)
        return(self.lag_confidences)

    def plot_response(self,newfig=True):
        if newfig is True:
            plt.close()
            fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.lag_grid,self.response)
        ax1.set_xlabel('response time (days)')
        ax1.set_ylabel('response function')

    def plot_timeseries(self,newfig=True):
        if newfig is True:
            plt.close()
            fig = plt.figure()
        ax1 = fig.add_subplot(111)
        if self.dates_global is None:
            ax1.plot(np.concatenate([self.result,self.predictions]),label='auto regressive model')
        else:
            ax1.plot(self.dates_global,np.concatenate([self.result, self.predictions]),
                     label='auto regressive model')
        ax1.plot(self.response_timeseries,label='data')
        plt.legend()
        ax1.set_xlabel('time (days)')
        ax1.set_ylabel('response time series')

    def plot_components(self,newfig = True):
        if newfig is True:
            plt.close()
            fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(self.A,aspect='auto')
        ax1.set_ylabel('component number')
        ax1.set_xlabel('time (days)')
        ax1.set_yticks(np.arange(self.ncomponents))

    def plot_covariance(self,newfig=True):
        if newfig is True:
            plt.close()
            fig = plt.figure()
        ax1 = fig.add_subplot(111)
        fn_plot_covariance(fig,ax1, self.cov, xlabel='', xticks=None)


    def presentation_plot(self,newfig = True):
        if newfig is True:
            plt.close()
            fig = plt.figure()
        y = []
        x = []
        ev = []
        title = []
        response = []
        xreal = []
        responses = self.get_ar_response()


        if self.fourier_component_results is not None:
            x.append(self.fourier_component_results)
            y.append(self.response_timeseries)
            title.append('self')
            ev.append(self.fourier_explained_variances)
            response.append([])
            xreal.append([])

        if self.trend_component_results is not None:
            x.append(self.trend_component_results)
            y.append(self.response_timeseries)
            title.append('trend')
            ev.append(self.trend_explained_variances)
            response.append([])
            xreal.append([])
        for i in range(len(self.ar_component_results)):
            x.append(self.ar_component_results[i])
            y.append(self.response_timeseries)
            title.append(self.labels_ar[i])
            ev.append(self.ar_explained_variances[i])
            response.append(responses[i])
            xreal.append(self.input_ar_timeseries[i])
        for i in range(len(self.idx_non_ar)):
            x.append(self.nonar_component_results[i])
            y.append(self.response_timeseries)
            ev.append(self.nonar_explained_variances[i])
            title.append(self.labels_non_ar[i])
            response.append([])
            xreal.append([])
        nplots = len(x)

        t = self.dates_global
        for i in range(nplots):
            # correlation figures
            ax1 = fig.add_subplot(4,nplots,i+1)
            ax1,fig = correlation_figure(x[i], y[i], 0,
                            order=1, xylabs=(title[i], 'flows (MT)'), figure_title='',
                            bin=True,
                            axfig=(ax1,fig),
                            global_title='correlation '+title[i]+' EV='+np.str(np.round(ev[i],2)))
            ax1.tick_params(axis='x', rotation=35)

            # time series figures
            ax2 = fig.add_subplot(4,nplots,nplots + i+1)
            ax2.plot(t,x[i],label = 'transformed '+ title[i])

            if len(xreal[i]) > 0:
                ax2b = ax2.twinx()
                ax2b.plot(t,xreal[i],color='r',label = 'input ' + title[i])
                ax2.legend(fontsize='xx-small',loc=3)
                ax2b.legend(fontsize='xx-small',loc=4)

            ax2.tick_params(axis='x', rotation=35)
            #response figures
            if len(response[i]) > 0:
                axr = fig.add_subplot(4,nplots,2*nplots + i+1)
                axr.plot(self.lag_grid,response[i])
                axr.set_ylabel(title[i] + ' response')
        #final result
        ax3 = fig.add_subplot(4,1,4)
        ax3.plot(t,self.response_timeseries+self.mean_response_timeseries,label='data')
        ax3.plot(t,self.result,label='model')
        self.calc_error_envelopes()
        ax3.fill_between(t,self.error_envelopes[:,0],self.error_envelopes[:,2],
                         alpha = 0.5,label = 'confidences',color='y')
        if self.npredict > 0:
            ax3.plot(self.dates_prediction,self.predictions,label='predictions')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=35)








    def plot_summary(self,newfig = True):
        if newfig is True:
            plt.close()
            fig = plt.figure()
        responses = self.get_ar_response()
        lag_confidences = self.get_lags()
        

        idx = 0
        ndown = len(self.idx_non_ar_count) + len(responses) + 1
        nalong = 2


        t = self.dates_global

        #plot fourier components
        if self.n_fourier > 0:
            ndown = ndown + 1
            ax1 = fig.add_subplot(ndown, nalong, 1)
            ax1.set_xlabel('time (days)')
            ax1.set_ylabel('fourier components')
            timeseries = self.fourier_component_results
            ax1.plot(t,timeseries,label='fourier components')
            ev = self.fourier_explained_variances
            title = 'Explained Variance: ' + np.str(np.round(ev, 2))
            ax1.set_title(title)
            ax1.tick_params(axis='x', rotation=45)
            #power spectrum
            ax1 = fig.add_subplot(ndown, nalong, 2)
            idf = np.array(self.idx_nonar_fourier)
            sin_amps = self.response[idf][::2]
            cos_amps = self.response[idf][1::2]
            pspec = sin_amps**2 + cos_amps**2
            frequencies = 1./np.array(self.periods)
            ax1.plot(frequencies,pspec)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_title('power spectrum')
            ax1.set_xlabel('frequency (cycles/ day)')
            ax1.set_ylabel('P(f)')
            idx = idx + 1


        #plot autoregressive components
        for i in range(len(responses)):
            ax1 = fig.add_subplot(ndown,nalong,idx*nalong+2)
            fn_plot_response(ax1,self.lag_grid,responses[i])
            lag = np.array(lag_confidences[i],dtype='float')
            title = 'Lags: '+np.str(np.round(lag[0],2))+\
                    ' < '+np.str(np.round(lag[1],2))+\
                    ' < '+np.str(np.round(lag[2],2))+' days '
            ax1.set_title(title)
            ax1 = fig.add_subplot(ndown, nalong, idx * nalong + 1)
            ts1 = self.input_ar_timeseries[i]
            ts2 = self.ar_component_results[i]
            ax1.plot(t,(ts1 - np.mean(ts1)) / np.std(ts1), label=self.labels_ar[i])
            ax1.plot(t,(ts2 - np.mean(ts2)) / np.std(ts2), label='post-convolution')
            ev = self.ar_explained_variances[i]
            title = 'Explained Variance: ' + np.str(np.round(ev, 2))
            ax1.set_title(title)
            ax1.set_xlabel('time (days)')
            ax1.set_ylabel(self.labels_ar[i])
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend(fontsize='xx-small')
            idx = idx + 1
        #plot non auto regressive components
        for i in range(len(self.idx_non_ar)):
            ax1 = fig.add_subplot(ndown, nalong, idx * nalong + 1)
            timeseries = self.A[self.idx_non_ar[i],:]
            ax1.plot(t,timeseries, label=self.labels_non_ar)
            ev = self.component_explained_variances[self.idx_non_ar[i]]
            title = 'Explained Variance: ' + np.str(np.round(ev, 2))
            ax1.set_title(title)
            ax1.set_xlabel('time (days)')
            ax1.set_ylabel(self.labels_non_ar[i])
            ax1.tick_params(axis='x', rotation=45)
            idx = idx + 1
        #plot the final result
        ax1 = fig.add_subplot(ndown,1,ndown)
        ax1.plot(t,self.response_timeseries,label='desired result',color='k')
        y = self.result#np.concatenate([self.result,self.predictions])
        ax1.plot(t,y,label='model',color='b')
        if self.error_envelopes is not None:
            eelo = self.error_envelopes[:len(t),0]
            eehi = self.error_envelopes[:len(t),2]
            ax1.fill_between(t,eelo,eehi,alpha=0.2,
                             label='model confidence limits',color='b')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(fontsize='xx-small')



    def fit_statistics(self):
        print('degrees of freedom:', np.round(self.degrees_of_freedom,2))
        print('sum of square residuals:',self.sum_of_square_residuals)
        print('AIC:',np.round(self.AIC,2))
        print('BIC:',np.round(self.BIC,2))
        print('explained variance:',np.round(self.explained_variance))




    def optimize_regularisation(self,defaults = np.linspace(1,1.e4,100),verbose = False):
        aic_optimize = []
        bic_optimize = []
        A_recalc = True

        idx_reg = np.arange(self.ncomponents)#np.arange(self.n_ar)
        for i in range(len(defaults)):
            self.regularize_weight = np.zeros(self.ncomponents)
            self.regularize_weight[idx_reg] = defaults[i]
            self.fit(A_recalc=A_recalc)
            self.response_timeseries = self.response_timeseries + self.mean_response_timeseries
            aic_optimize.append(self.AIC)
            bic_optimize.append(self.BIC)
            A_recalc = False
        self.regularize_weight[idx_reg] = defaults[np.argmin(aic_optimize)]

        if verbose is True:
            print('optimising regularisation parameter...')
            print('regularisation weight = ', defaults[i])
            self.fit_statistics()
            print('ncomponents',self.ncomponents)
            print(idx_reg)
            print('setting regularize weight to ', defaults[np.argmin(aic_optimize)])


    def iterated_optimize_regularisation(self,iterations = 4, ntrials = 100,verbose = False):
        trials = np.linspace(1, 1.e4, ntrials)
        for i in range(iterations):
            self.optimize_regularisation(defaults = trials)
            rw_now = np.max(self.regularize_weight)
            idx = np.where(trials == rw_now)[0][0]
            idlo = max(0, idx-1)
            idhi = min(len(trials)-1,idx+1)
            trials = np.linspace(trials[idlo],trials[idhi],ntrials)
            if verbose is True:
                print('optimizing regularisation...')
                print('iteration: ',i+1,' of ',iterations,' regularisation =',np.round(rw_now,2))
                print(iterations, ' iterations. regularisation =', np.round(rw_now, 2))

    def calc_error_envelopes(self,nsamples = 1000, Acustom = None,verbose = False):
        if Acustom is None:
            Afinal = self.A
        else:
            Afinal = Acustom
        n = np.shape(Afinal)[1]
        response_samples = np.random.multivariate_normal(self.response,self.cov,size=nsamples)
        results_samples = np.zeros((n,nsamples))
        for i in range(nsamples):
            results_samples[:,i] = np.dot(response_samples[i,:],Afinal) + self.mean_response_timeseries
        self.error_envelopes = np.percentile(results_samples,self.confidence_limits,axis=1).T

        if verbose is True:
            plt.close()
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            #for i in range(nsamples):
            #    ax1.plot(results_samples[:,i],alpha=0.1,color='r',label='samples')
            ax1.plot(self.error_envelopes[:,0],color='y',label='envelopes')
            ax1.plot(self.error_envelopes[:,1],color='y',label=None)
            ax1.plot(self.error_envelopes[:,2],color='y',label=None)
            ax1.plot(self.result,color='b',label='result')
            ax1.set_xlabel('time')
            ax1.set_ylabel('flux')
            plt.legend()
            plt.show()




    def predict(self,nsteps=10):
        #pad the orriginal time series
        self.npredict = nsteps
        ts_ar_pad = []
        for ts in self.input_ar_timeseries:
            ts_ar_pad.append( np.pad(ts,(0,nsteps),mode='constant',constant_values=(ts[0],ts[-1])) )
        ts_nonar_pad = []
        for ts in self.input_nonar_timeseries:
            ts_nonar_pad.append( np.pad(ts,(0,nsteps),mode='constant',constant_values=(ts[0],ts[-1])) )
        Afinal = make_A(ts_ar_pad,ts_nonar_pad,self.lagrange, self.ndata+nsteps)
        #use the correct extrapolation for the trend and fourier time series
        t = np.arange(self.ndata + nsteps)
        for i in range(len(self.idx_nonar_trend)):
            Afinal[self.idx_nonar_trend[i],:] = t**(i+1)
        idx = 0
        for i in range(len(self.idx_nonar_fourier)):
            p = self.periods[idx]
            if np.mod(i,2) == 0:
                x = np.sin(2 * np.pi / p * t)
            else:
                x = np.cos(2 * np.pi / p * t)
                idx = idx + 1
            Afinal[self.idx_nonar_fourier[i],:] = x
        self.predictions = np.dot(self.response,Afinal)[-nsteps:] + self.mean_response_timeseries
        self.calc_error_envelopes(Acustom = Afinal)
        if type(self.dates_global) is np.ndarray:
            self.dates_prediction = np.arange(self.ndata,self.ndata + nsteps)
        else:
            start = self.dates_global.values[-1]+pd.Timedelta(1,unit='D')
            self.dates_prediction = pd.date_range(start=start,periods = nsteps)



    def evaluate_model(self,y_predicted,y_true):
        self.mape, self.rms, self.percent_agree, self.rms_mean, self.mad = \
            evaluate_model(y_predicted, y_true, verbose=True)

if __name__ == '__main__':
    font = {'family': 'normal',
            #'weight': 'bold',
            'size': 6}
    import matplotlib
    matplotlib.rc('font',**font)

    lag = 30.0
    wide = 1.0

    n = 400
    time = np.arange(n)
    signal = 10
    noise  = 0.2
    season = 0.000025 * time ** 2
    npredict = 100
    y    = signal*np.sin(2*np.pi*time/90) + noise*np.random.randn(n)


    # construct mock time series and include impulse response lag function
    tau      = np.arange(np.int(-n/2)+1,np.int(n/2)-1,1)
    response = np.exp(-0.5 * (tau - lag) ** 2 / wide ** 2) / (2 * np.pi * wide ** 2)
    echo     = scipy.convolve(y,response,mode='same')
    a = RLI()
    a.lagrange = [-30, 110]
    drift_model = time**2


    a.response_timeseries = echo + season
    a.response_timeseries = (a.response_timeseries - np.mean(a.response_timeseries))/np.std(a.response_timeseries)
    today = pd.datetime.now()
    dates = pd.date_range(start=today, periods=n)

    a.add_ar_component((y - np.mean(y))/np.std(y), name='driver auto regressor',dates=dates)
    #a.add_fourier_components(periods = np.arange(1,500))
    a.add_trend(order=2)

    #weights can either be a single number or a numpy array with different
    #regularisation weights for each parameter
    weights = 100.
    a.regularize_weight = weights
    a.iterated_optimize_regularisation()
    a.fit()
    a.fit_statistics()
    a.predict(nsteps=npredict)
    a.plot_summary()
    plt.tight_layout()
    plt.savefig('/Users/david/Desktop/test_RLI_figs.pdf')
    a.presentation_plot()
    plt.tight_layout()
    plt.show()

    a.plot_summary()
    plt.show()


