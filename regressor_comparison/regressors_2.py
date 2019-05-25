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
        self.n_informative = 5
        self.alphas = np.arange(0.1,10.,0.1)

    def make_fake(self):

        #make fake data
        self.X,self.y = sklearn.datasets.make_regression(n_samples=self.N,
                                         n_features=self.K,
                                         n_informative=self.n_informative,
                                         n_targets=1,
                                         bias=0.0,
                                         effective_rank=None,
                                         tail_strength=0.5, noise=0.0,
                                         shuffle=True, coef=False, random_state=None)
        self.y = np.sum(self.X[:, :self.n_informative], axis=1)



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
               linear_model.RidgeCV(alphas = self.alphas,store_cv_values= True),
               linear_model.LassoCV(alphas=self.alphas),
               GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=0)],
               'name':['lassolars','rfr','etr','gbr','mpl','glm','Ridge CV','Lasso CV','gpr']}


    def fit_1round(self):

        output = {'name':self.fits['name'],
                  'score':[],
                  'direction':[],
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

            dp = p[1:] - self.ytest[:-1]
            dt = self.ytest[1:] - self.ytest[:-1]
            dpdt = dp*dt
            idgood = np.where(dpdt > 0)[0]
            output['direction'].append(100*np.float(len(idgood))/len(dp))
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
            x = self.output[['score','train time','predict time','direction']].values[:,:]
            if i == 0:
                scores =     x[:,0]
                train_time = x[:,1]
                test_time =  x[:,2]
                direction = x[:,3]
            else:
                scores = scores + x[:,0]
                train_time = train_time + x[:,1]
                test_time = test_time + x[:,2]
                direction = direction + x[:,3]
        self.average_output = pd.DataFrame.copy(self.output)
        self.average_output['score'] = scores/nrounds
        self.average_output['train time'] =train_time/nrounds
        self.average_output['test time'] = test_time/nrounds
        self.average_output['direction'] = direction/nrounds
        self.average_output = self.average_output.sort_values(by='score')


    def plot_parms(self):
        '''
        plot the parameter values for different methods
        want sharp transition from 1 to zero as we go from important to
        irrelevant parameters
        :return:
        '''
        plt.close()
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_xlabel('parameter number')
        ax1.set_ylabel('parameter value')
        ax2 = fig.add_subplot(212)
        ax2.set_xlabel('time')
        ax2.set_ylabel('y')
        nm = len(self.fits['model'])
        ax2.plot(self.y,label='true',color='k')
        for ir in range(nm):
            try:
                mname = self.fits['name'][ir]
                m = self.fits['model'][ir]
                parms = m.coef_
                pred = m.predict(self.X)
                ax1.plot(parms,label=mname)
                ax2.plot(pred, label=mname)
            except:
                pass
        plt.legend()






if __name__ == '__main__':
    nave = 10
    a = test_regressors()
    a.alphas = np.logspace(-13,-8,100)
    a.N = 1000
    a.K = 800
    a.make_fake()
    a.set_regressors()
    a.fit_multiround(nrounds = nave)
    print('average outouts after',nave,'iterations')
    print(a.average_output)
    a.plot_parms()
    plt.show()


    check = ['Ridge CV','Lasso CV']
    for c in check:
        idr = a.fits['name'].index(c)
        r = a.fits['model'][idr]
        r_cv = pd.DataFrame(
            {'alpha':a.alphas,'score':np.mean(r.cv_values_,axis=0)}
        )
        plt.plot(r_cv['alpha'],r_cv['score'])
        plt.xscale('log')
        plt.yscale('log')
        plt.show()





