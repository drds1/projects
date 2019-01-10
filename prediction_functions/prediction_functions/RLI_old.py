import numpy as np
import pandas as pd
import linvpy.linvpy as lp
import scipy
import matplotlib.pylab as plt
import prediction_functions.signal_decomp as dcomp
from prediction_functions.evaluate_model import *


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
    print(np.mean(cor),np.min(cor),np.max(cor))
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
    

def make_A(timeseries_ar,timeseries_nonar,lagrange):
    nepoch = len(timeseries_ar[0])
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
        for i in range(lenlag):
            A[i, :] = np.roll(A[i, :], idlag[i])
        A_out = np.vstack((A_out,A))

    idx_nonar = np.arange(np.shape(A_out)[0],np.shape(A_out)[0]+n_nonar)
    for i in range(n_nonar):
        ts = timeseries_nonar[i]
        A  = np.zeros((1,nepoch))
        A[0,:] = ts
        A_out = np.vstack((A_out,A))
    return(A_out,idx_nonar)


#recast driving time series into N X Nlag matrix A for linvpy process
class RLI:
    #input parameters
    response_timeseries = 0
    lagrange = []#[-20,20]
    response = 0
    result = 0
    A = None
    regularize = 'minimize squares'
    regularize_weight = 0
    idx_response = []
    confidence_limits = [16,50,84]
    idx_non_ar = []
    idx_ar = []
    labels_ar = []
    labels_non_ar = []
    predictions = np.array([])
    input_ar_timeseries = []
    input_nonar_timeseries = []

    def add_component(self,timeseries,name=None):
        n = np.shape(timeseries)[0]
        if self.lagrange == []:
            idlo = np.int(-n/2) + 2
            idhi = np.int(n/2) - 2
            self.lagrange=[idlo,idhi]
        if self.A is not None:
            self.A = np.vstack([self.A,timeseries])
        else:
            A = np.zeros((1,len(timeseries)))
            A[0,:] = timeseries
            self.A = A
        self.ncomponents, self.ndata = np.shape(self.A)
        #remember which components are or are not autoregressive
        self.idx_non_ar.append(np.shape(self.A)[0]-1)
        if name is None:
            self.labels_non_ar.append('Non AR \ncomponent '+np.str(len(self.idx_non_ar)))
        else:
            self.labels_non_ar.append(name)

    def add_ar_component(self,timeseries, name = None):
        n = np.shape(timeseries)[0]
        if self.lagrange == []:
            idlo = np.int(-n/2) + 2
            idhi = np.int(n/2) - 2
            self.lagrange=[idlo,idhi]
        idlo, idhi = self.lagrange
        idlag = np.arange(idlo, idhi + 1, 1)
        self.lag_grid = idlag
        lenlag = len(idlag)
        A = np.tile(timeseries, lenlag)
        A = np.reshape(A, (lenlag, n))
        for i in range(lenlag):
            A[i,:] = np.roll(A[i,:],idlag[i])
        if self.A is not None:
            self.A = np.vstack([self.A,A])
        else:
            self.A = A
        self.ncomponents, self.ndata = np.shape(self.A)
        self.idx_ar.append(self.ncomponents-1)
        if name is None:
            self.labels_ar.append('AR \ncomponent '+np.str(len(self.idx_ar)))
        else:
            self.labels_ar.append(name)

        #for the lag calculation,
        #remember the weight indicees belonging to each auto regressive time series
        idhi = np.shape(self.A)[0]
        idlo = idhi - lenlag
        self.idx_response.append([idlo,idhi])

    def fit(self):
        #perform fit
        parms, cov, r2, mse, importance = \
            dcomp.constituents_fit(self.response_timeseries, self.A.T,
                                   regularize=self.regularize,
                                   regularize_weight=self.regularize_weight)
        #tau = lp.TauEstimator(loss_function=lp.Bisquare, lamb=1.0)
        #parms = tau.estimate(self.A.T, self.response_timeseries)[0]
        self.response = parms
        #self.response[self.response <= 0 ] = 0
        self.cov = cov
        self.result = np.dot(self.response,self.A)

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

        self.ar_component_results   = []
        self.ar_component_residuals = []
        self.ar_explained_variances = []
        for i in range(len(self.idx_ar)):
            sl = slice(self.idx_response[i][0],self.idx_response[i][1])
            self.ar_component_results.append( np.dot(self.response[sl],self.A[sl,:]) )
            self.ar_component_residuals.append( self.ar_component_results[i] - self.response_timeseries )
            self.ar_explained_variances.append( 1. -
                                               np.var(self.ar_component_residuals[i])/
                                               self.variance_response_timeseries )

    def get_ar_response(self):
        self.ar_response = []
        for lag_limits in self.idx_response:
            self.ar_response.append( self.response[lag_limits[0]:lag_limits[1]] )
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
        ax1.plot(np.concatenate([self.result,self.predictions]),label='auto regressive model')
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

    def plot_summary(self,newfig = True):
        if newfig is True:
            plt.close()
            fig = plt.figure()
        responses = self.get_ar_response()
        lag_confidences = self.get_lags()
        idx = 0
        ndown = len(self.idx_non_ar) + len(responses) + 1
        nalong = 2
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
            ts1 = self.A[self.idx_ar[i], :]
            ts2 = self.ar_component_results[i]
            ax1.plot((ts1 - np.mean(ts1))/np.std(ts1), label=self.labels_ar[i])
            ax1.plot((ts2 - np.mean(ts2))/np.std(ts2), label='post-convolution')
            ev = self.ar_explained_variances[i]
            title = 'Explained Variance: ' + np.str(np.round(ev, 2))
            ax1.set_title(title)
            ax1.set_xlabel('time (days)')
            ax1.set_ylabel(self.labels_ar[i])
            ax1.legend(fontsize='xx-small')
            idx = idx + 1
        #plot non auto regressive components
        for i in range(len(self.idx_non_ar)):
            ax1 = fig.add_subplot(ndown, nalong, idx * nalong + 1)
            timeseries = self.A[self.idx_non_ar[i],:]
            ax1.plot(timeseries,label=self.labels_non_ar)
            ev = self.component_explained_variances[self.idx_non_ar[i]]
            title = 'Explained Variance: ' + np.str(np.round(ev, 2))
            ax1.set_title(title)
            ax1.set_xlabel('time (days)')
            ax1.set_ylabel(self.labels_non_ar[i])
            idx = idx + 1
        #plot the final result
        ax1 = fig.add_subplot(ndown,1,ndown)
        ax1.plot(self.response_timeseries,label='desired result')
        ax1.plot(np.concatenate([self.result,self.predictions]),label='model')
        ax1.legend()



    def fit_statistics(self):
        print('degrees of freedom:', np.round(self.degrees_of_freedom,2))
        print('sum of square residuals:',self.sum_of_square_residuals)
        print('AIC:',np.round(self.AIC,2))
        print('BIC:',np.round(self.BIC,2))
        print('explained variance:',np.round(self.explained_variance))




    def optimize_regularisation(self,defaults = np.linspace(1,1.e4,100),verbose = False):
        aic_optimize = []
        bic_optimize = []
        for i in range(len(defaults)):
            self.regularize_weight = np.ones(self.ncomponents)*defaults[i]
            if len(self.idx_non_ar) > 0:
                self.regularize_weight[np.array(self.idx_non_ar,dtype='int')] = 0
            self.fit()
            aic_optimize.append(self.AIC)
            bic_optimize.append(self.BIC)
        self.regularize_weight = np.ones(self.ncomponents)*defaults[np.argmin(aic_optimize)]
        if len(self.idx_non_ar) > 0:
            self.regularize_weight[np.array(self.idx_non_ar, dtype='int')] = 0
        if verbose is True:
            print('optimising regularisation parameter...')
            print('regularisation weight = ', defaults[i])
            self.fit_statistics()
            print('setting regularize weight to ', defaults[np.argmin(aic_optimize)])
            print('non auto regressive components exempted...', self.idx_non_ar)

    def iterated_optimize_regularisation(self,iterations = 4, ntrials = 100,verbose = False):
        trials = np.linspace(1, 1.e4, ntrials)
        print('optimizing regularisation...')
        for i in range(iterations):
            self.optimize_regularisation(defaults = trials)
            rw_now = np.max(self.regularize_weight)
            idx = np.where(trials == rw_now)[0][0]
            idlo = max(0, idx-1)
            idhi = min(len(trials)-1,idx+1)
            trials = np.linspace(trials[idlo],trials[idhi],ntrials)
            if verbose is True:
                print('iteration: ',i+1,' of ',iterations,' regularisation =',np.round(rw_now,2))
        print(iterations, ' iterations. regularisation =', np.round(rw_now, 2))

    def predict(self,nsteps=10):
        Afinal = np.reshape(np.tile(a.A[:, -1], nsteps), (nsteps, a.ncomponents)).T
        Aextend = np.hstack((self.A,Afinal))
        self.predictions = np.dot(self.response,Aextend)[-nsteps:]

    def evaluate_model(self,y_predicted,y_true):
        self.mape, self.rms, self.percent_agree, self.rms_mean, self.mad = \
            evaluate_model(y_predicted, y_true, verbose=True)

if __name__ == '__main__':
    lag = 30.0
    wide = 1.0

    n = 200
    time = np.arange(n)
    signal = 10
    noise  = 0.2
    season = 0.01 * time ** 2
    npredict = 10
    y    = signal*np.sin(2*np.pi*time/90) + noise*np.random.randn(n) + season
    y    = (y - np.mean(y))/np.std(y)


    # construct mock time series and include impulse response lag function
    tau      = np.arange(np.int(-n/2)+1,np.int(n/2)-1,1)
    response = np.exp(-0.5 * (tau - lag) ** 2 / wide ** 2) / (2 * np.pi * wide ** 2)
    echo     = scipy.convolve(y,response,mode='same')
    echo = (echo - np.mean(echo))/np.std(echo)
    a = RLI()
    a.lagrange = [-50, 90]
    drift_model = time**2

    a.add_ar_component(y[:-npredict],name='self auto regressor')
    a.response_timeseries = echo[:-npredict]
    a.add_component(drift_model[:-npredict],name = 'drift')

    #weights can either be a single number or a numpy array with different
    #regularisation weights for each parameter
    weights = 1
    a.regularize_weight = weights
    a.iterated_optimize_regularisation()
    a.fit()
    a.fit_statistics()
    a.predict(nsteps=npredict)
    a.evaluate_model(a.predictions,echo[-npredict:])
    #a.plot_response()
    #plt.show()
    #a.plot_timeseries()
    #plt.show()
    a.plot_summary()
    plt.tight_layout()
    plt.show()

    a.plot_covariance()
    plt.show()

    #code diagnostic
    #response_true = np.exp(-0.5 * (a.lag_grid - lag) ** 2 / wide ** 2) / (2 * np.pi * wide ** 2)
    #plt.plot(a.lag_grid,response_true,label='True response')
    #plt.plot(a.lag_grid,a.response[:],label='inferred response')
    #plt.legend()
    #plt.show()

    #a.plot_components()
    #plt.show()