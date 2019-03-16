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


    def add_covariates(self,covariates = None,importances = None):
        '''
        make the fake data. Either enter custom covariates or make random ones
        :return:
        '''
        self.extra_covariates = covariates
        self.covariates = np.shape(covariates[0,:])[0]
        for i in range(self.covariates):
            if covariates is None:
                self.extra_covariates[:,i] = np.random.randn(self.npoints)
            if importances is None:
                a = 1.0
            else:
                a = importances[i]
            self.driver = self.driver + a*self.extra_covariates[:,i]

    def add_driver_seasonality(self,periods,amplitudes = None):

        idx = 0
        for p in periods:
            if amplitudes is None:
                a = 1.0
            else:
                a = amplitudes[idx]
            self.driver = self.driver + a*np.sin(2*np.pi*self.t/p)
            idx = idx + 1

    def add_driver_trend(self,order,amplitudes):

        idx = 0
        for o in order:
            if amplitudes is None:
                a = 0
            else:
                a = amplitudes[idx]
            self.driver = self.driver + a*((2.*self.t)/self.npoints - 1.)**o



if __name__ == '__main__':
    a = fake_data()
    a.covariates = 4
    a.npoints = 100
    a.initialise()
    a.add_covariates()
    a.add_driver_trend(order = [2.])
    a.add_driver_seasonality(periods =  [5.,13.])

