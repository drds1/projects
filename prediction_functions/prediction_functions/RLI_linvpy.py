import numpy as np
import pandas as pd
import linvpy.linvpy as lp
import scipy
import matplotlib.pylab as plt


#recast driving time series into N X Nlag matrix A for linvpy process
class RLI:
    #input parameters
    driving_timeseries  = 0
    response_timeseries = 0
    y = 0
    lagrange = [-20,20]
    response = 0
    result = 0
    extra_components = False


    def fit(self):
        idlo,idhi = self.lagrange
        idlag = np.arange(idlo,idhi+1,1)
        lenlag = len(idlag)
        ndrive = np.shape(self.driving_timeseries)[0]
        A = np.tile(self.driving_timeseries,lenlag)
        A = np.reshape(A,(lenlag,ndrive))
        for i in range(lenlag):
         A[i,:] = np.roll(A[i,:],idlag[i])
        A = A.T

        len_extra_components = np.shape(self.extra_components[0,:])[0]

        print(np.shape(A), np.shape(self.extra_components))
        if self.extra_components is not None:
            A = np.hstack([self.extra_components,A])

        tau = lp.TauEstimator(loss_function=lp.Bisquare, lamb=1.0)
        self.response = tau.estimate(A, self.response_timeseries)[0]

        if self.extra_components is None:
            self.result = scipy.convolve(self.driving_timeseries,self.response,mode='same')
        else:
            self.result = scipy.convolve(self.driving_timeseries, self.response[len_extra_components:],
                                         mode='same') \
                          + np.sum(self.response[:len_extra_components]*self.extra_components,axis=1)
        print(scipy.convolve(self.driving_timeseries, self.response[len_extra_components:],
                                         mode='same'))



    def plot_response(self):
        plt.close()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.response)
        ax1.set_xlabel('response time (days)')
        ax1.set_ylabel('response function')

    def plot_timeseries(self):
        plt.close()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.result,label='auto regressive model')
        ax1.plot(self.response_timeseries,label='data')
        plt.legend()
        ax1.set_xlabel('time (days)')
        ax1.set_ylabel('response time series')





if __name__ == '__main__':
    lag = 30.0
    wide = 5.0

    n = 200
    time = np.arange(n)
    signal = 10
    noise  = 0.2
    y    = signal*np.sin(2*np.pi*time/90) + noise*np.random.randn(n)
    # construct mock time series and include impulse response lag function
    tau      = np.arange(np.int(-n/2)+1,np.int(n/2)-1,1)
    response = np.exp(-0.5 * (tau - lag) ** 2 / wide ** 2) / (2 * np.pi * wide ** 2)
    echo     = scipy.convolve(y,response,mode='same')

    a = RLI()
    a.driving_timeseries  = y
    a.response_timeseries = echo
    a.lagrange            = [-1,1]

    ec = np.ones((n,1))
    ec[:,0] = echo
    a.extra_components    = ec
    a.fit()
    a.plot_response()
    plt.show()
    a.plot_timeseries()
    plt.show()


    #from pylab import *
#
    #plot(dat[:, 0], dat[:, 1])
    #plot(dat[:, 0], echo)
    #savefig('test_timeseries.png')
#
    #data = np.zeros((n, 2))
    #data[:, 0] = dat[:, 1]
    #data[:, 1] = echo
    #data = data[::10, :]
#
    #X, y = data[:, 0], data[:, 1]
    #data = pd.DataFrame(data, columns=['drive', 'echo'])
