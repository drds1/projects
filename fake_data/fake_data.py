import numpy as np
import pandas as pd

class fake_data:

    def __init__(self):
        self.driver_periods =[]
        self.driver_trend  = []
        self.covariates = 4
        self.npoints = 100

    def initialise(self):
        self.extra_covariates = np.zeros((self.npoints, self.covariates))
        self.driver = np.zeros(self.npoints)
        self.t = np.arange(self.npoints)


    def add_covariates(self,covariates = None,importances = None,iseed = None):
        '''
        make the fake data. Either enter custom covariates or make random ones
        :return:
        '''
        if iseed is not None:
            np.random.seed(iseed)
        if covariates is None:
            for i in range(self.covariates):
                self.extra_covariates[:,i] = np.random.randn(self.npoints)
        else:
            self.extra_covariates = covariates

        self.covariates = np.shape(self.extra_covariates[0, :])[0]
        if importances is None:
            a = np.ones(self.covariates)
        else:
            a = importances

        for i in range(self.covariates):
            self.driver = self.driver + a[i]*self.extra_covariates[:,i]


    def add_driver_seasonality(self,periods,amplitudes = None):

        idx = 0
        for p in periods:
            if amplitudes is None:
                a = 1.0
            else:
                a = amplitudes[idx]
            self.driver = self.driver + a*np.sin(2*np.pi*self.t/p)
            idx = idx + 1

    def add_driver_trend(self,order,amplitudes=None):

        idx = 0
        for o in order:
            if amplitudes is None:
                a = 1.0
            else:
                a = amplitudes[idx]
            self.driver = self.driver + a*((2.*self.t)/self.npoints - 1.)**o



if __name__ == '__main__':
    a = fake_data()
    a.covariates = 4
    a.npoints = 200
    a.initialise()
    a.add_covariates(importances=[1.,0.1,0.1,0.1],iseed = 343435)


