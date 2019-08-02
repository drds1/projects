import statsmodels.tsa.arima_process as st
import numpy as np
from sklearn import linear_model
import pandas as pd

def generate_arma(p=[1.0,0.75, -0.25],
                  q=[1.0,0.65,0.35],
                  n=100):
    '''
    generate and return n samples of an ARMA(p,q) process
    :param p:
    :param q:
    :param n:
    :return:
    '''
    x = st.arma_generate_sample(np.array(p),np.array(q),nsample=n)
    return(x)




class fit_arma:

    def __init__(self):
        self.y = None
        self.p = 3
        self.q = 3
        self.Niterations = 10000
        self.model = linear_model.LinearRegression(fit_intercept=False)
        self.coefs = None

    def make_ar(self):
        '''
        roll the array to simulate ar terms
        :return:
        '''
        self.N = len(self.y)
        self.Xar = np.zeros((self.N,self.p))
        for i in range(1,self.p+1):
            x = np.roll(self.y,-i)
            self.Xar[:,i-1] = x


    def make_ma(self):
        '''
        make a 'moving average array using gaussian random numbers
        :return:
        '''
        r = np.random.randn(self.N)
        self.Xma = np.zeros((self.N,self.q))
        for i in range(1,self.q+1):
            x = np.roll(r,-i)
            self.Xma[:,i-1] = x


    def fit_1it(self,X,y):
        '''
        fit the linear model
        :return:
        '''
        self.model.fit(X,y)
        self.coefs = self.model.coef_

    def fit(self):
        '''
        fit in iterations
        :return:
        '''
        Npar = self.p + self.q
        self.output_coefs = np.zeros((Npar,self.Niterations))
        self.make_ar()
        for i in range(self.Niterations):
            self.make_ma()
            X = np.hstack((self.Xar,self.Xma))
            self.fit_1it(X,self.y)
            self.output_coefs[:,i] = self.coefs

    def get_output(self):
        '''
        output parameters in sensible format
        :return:
        '''
        self.output = pd.DataFrame(self.output_coefs).transpose()
        cols = []
        for i in range(self.p):
            cols.append('AR '+str(i+1))
        for i in range(self.q):
            cols.append('MA '+str(i+1))
        self.output.columns=cols

if __name__ == '__main__':
    x = generate_arma()
    f = fit_arma()
    f.y = x
    f.fit()
    f.get_output()
    op = f.output
    print(op.mean(axis=0))
    print(op.std(axis=0)**2)
    import corner
    import matplotlib.pylab as plt
    corner.corner(op)
    plt.show()
