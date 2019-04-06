from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import sys
sys.path.append("/Users/david/github_datascience/projects/fake_data/")
from sklearn import ensemble
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
import time
import numpy as np
import matplotlib.pylab as plt
from sklearn.tree import export_graphviz
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import sklearn.datasets
import pandas as pd
import time

K = 100
N = 2000


class test_regressors:

    def __init__(self):
        self.K = 100
        self.N = 2000

    def make_fake(self):

        #make fake data
        self.X,self.y = sklearn.datasets.make_regression(n_samples=N,
                                         n_features=K,
                                         n_informative=3,
                                         n_targets=1,
                                         bias=0.0,
                                         effective_rank=None,
                                         tail_strength=0.5, noise=0.0,
                                         shuffle=True, coef=False, random_state=None)


        idxtrain = int(0.7*K)
        self.Xtrain = self.X[:idxtrain,:]
        self.Xtest = self.X[idxtrain:,:]
        self.ytrain = self.y[:idxtrain]
        self.ytest = self.y[idxtrain:]



    def set_regressors(self):
        self.fits = {'model':[linear_model.LassoLars(),
               RandomForestRegressor(n_estimators=10, random_state=42),
               ExtraTreesRegressor(n_estimators=10, random_state=42),
               ensemble.GradientBoostingRegressor(),
               MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1),
               linear_model.LinearRegression(),
               GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=0)],
               'name':['lassolars','rfr','etr','gbr','mpl','glm','gpr']}


    def fit_1round(self):

        output = {'name':self.fits['name'],
                  'score':[],
                  'train time':[],
                  'predict time':[]}
        for i in range(len(self.fits['model'])):
            mnow = self.fits['model'][i]
            name = self.fits['name'][i]

            t1 = time.time()
            mnow.fit(self.Xtrain,self.ytrain)
            t2 = time.time()
            p = mnow.predict(self.Xtest)
            t3 = time.time()
            cisq_N = np.sum((p - self.ytest)**2)
            output['score'].append(cisq_N)
            output['train time'].append(t2-t1)
            output['predict time'].append(t3-t2)

        df = pd.DataFrame(output)

        self.output = df


    def fit_multiround(self,nrounds=10):
        print('averaging scores, compute times over',nrounds,'iterations')
        for i in range(nrounds):
            print('iteration',i)

            self.fit_1round()
            x = self.output[['score','train time','predict time']].values[:,:]
            if i == 0:
                scores =     x[:,0]
                train_time = x[:,1]
                test_time =  x[:,2]
            else:
                scores = scores + x[:,0]
                train_time = train_time + x[:,1]
                test_time = test_time + x[:,2]
        self.average_output = pd.DataFrame.copy(self.output)
        self.average_output['score'] = scores/nrounds
        self.average_output['train time'] =train_time/nrounds
        self.average_output['test time'] = test_time/nrounds
        self.average_output = self.average_output.sort_values(by='score')



if __name__ == '__main__':
    nave = 10
    a = test_regressors()
    a.make_fake()
    a.set_regressors()
    a.fit_multiround(nrounds = nave)
    print('average outouts after',nave,'iterations')
    print(a.average_output)





