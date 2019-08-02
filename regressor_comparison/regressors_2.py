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
        self.fits = {'model':[linear_model.LassoLars(alpha=3),
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
        #print('averaging scores, compute times over',nrounds,'iterations')
        for i in range(nrounds):
        #    print('iteration',i)

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




class test_model_Nk_ratio:

    def __init__(self):
        self.fits = {'model':
         [linear_model.LinearRegression(),
          linear_model.LassoCV(alphas=[0.1,1,10.], cv=3),
          linear_model.RidgeCV(alphas=[0.1,1,10.], cv=3)],
         'name':
         ['General Linear Model',
          'Lasso Regression',
          'Ridge Regression']}


        self.K = list(np.arange(2,100,1))
        self.N = [1000] * len(self.K)
        self.n_informative = [2] * len(self.K)

    def run(self):
        self.df_op = pd.DataFrame({})
        idx = 0
        for K in self.K:
            a = test_regressors()
            a.fits = self.fits
            a.n_informative = self.n_informative[idx]
            a.N = self.N[idx]
            a.K = K
            a.make_fake()
            a.fit_multiround(nrounds=5)
            df = a.average_output[['name','score','direction']]
            df['total components'] = [K]*len(df)
            df['useful components'] = [a.n_informative]*len(df)
            df['useless components'] = [K - a.n_informative]*len(df)
            df['useful-total ratio'] = df['useful components']/df['total components']
            df['useful-useless ratio'] = df['useful components']/df['useless components']
            self.df_op = self.df_op.append(df)
            idx = idx + 1


    def plot_results(self,
                     xvalue = 'useless components',
                     xlabel='Number of useless components',
                     global_title=''):
        '''
        plot the results (score vs number of total components
        :return:
        '''
        fig = plt.figure()
        st = fig.suptitle(global_title, fontsize="x-large")

        metrics = ['score','direction']
        title = ['L Error','direction']
        nm = len(metrics)

        for i in range(nm):
            ax1 = fig.add_subplot(1,nm,1+i)

            names = list(self.df_op['name'].unique())
            for n in names:
                dfnow = self.df_op[self.df_op['name'] == n]
                X = list(dfnow[xvalue])
                y = list(dfnow[metrics[i]])
                ax1.plot(X,y,label=n)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(title[i])
            ax1.set_title(title[i])
            plt.legend(fontsize='x-small')
        plt.tight_layout()
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)





if __name__ == '__main__':
    nave = 10
    a = test_regressors()
    alphas = np.logspace(-1, 1, 3)

    a.fits = {'model': [linear_model.LassoLars(alpha=3),
                           RandomForestRegressor(n_estimators=10, random_state=42),
                           ExtraTreesRegressor(n_estimators=10, random_state=42),
                           #ensemble.GradientBoostingRegressor(),
                           #MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1),
                           linear_model.LinearRegression(),
                           linear_model.RidgeCV(alphas=alphas, cv=3),
                           linear_model.LassoCV(alphas=alphas, cv=3),
                           GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=0)],
                 'name': ['lassolars',
                          'rfr',
                          'etr',
                          #'gbr',
                          #'mpl',
                          'glm',
                          'Ridge CV',
                          'Lasso CV',
                          'gpr']}



    a.N = 1000
    a.K = 800
    a.n_informative = 2
    a.make_fake()
    a.fit_multiround(nrounds = nave)
    #print('average outouts after',nave,'iterations')
    print(a.average_output)

    # number of useless parameters study
    x = test_model_Nk_ratio()
    x.K = list(np.arange(2, 100, 1))
    x.N = [1000] * len(x.K)
    x.n_informative = [2] * len(x.K)
    x.run()
    x.plot_results(xlabel = 'Number of useless components',
                   global_title='')
    plt.savefig('useless_absolute.pdf')
    #plt.show()


    # ratio of useful/useless components
    x = test_model_Nk_ratio()
    x.n_informative = np.arange(2,101)
    x.K = [100]*len(x.n_informative)
    x.N = [1000] * len(x.K)
    x.run()
    x.plot_results(xlabel='fraction of useful components',
                   xvalue='useful-total ratio')
    plt.savefig('useless_ratio.pdf')


    # ratio of useful/useless components
    x = test_model_Nk_ratio()
    x.n_informative = np.arange(2,11)
    x.K = [10]*len(x.n_informative)
    x.N = [1000] * len(x.K)
    x.run()
    x.plot_results(xlabel='fraction of useful components',
                   xvalue='useful-total ratio',
                   global_title='10 components')
    plt.savefig('useless_ratio_10cmpts.pdf')



    # ratio of useful/useless components
    x = test_model_Nk_ratio()
    x.n_informative = np.arange(2,101)
    x.K = [100]*len(x.n_informative)
    x.N = [10000] * len(x.K)
    x.run()
    x.plot_results(xlabel='fraction of useful components',
                   xvalue='useful-total ratio',
                   global_title='large sample ratio')
    plt.savefig('useless_ratio_large_sample.pdf')

    # ratio of useful/useless components
    x = test_model_Nk_ratio()
    x.n_informative = np.arange(2,101)
    x.K = [100]*len(x.n_informative)
    x.N = [100] * len(x.K)
    x.run()
    x.plot_results(xlabel='fraction of useful components',
                   xvalue='useful-total ratio',
                   global_title='small sample ratio')
    plt.savefig('useless_ratio_small_sample.pdf')








    #a.plot_parms()
    #plt.show()
#
#
    #check = ['Ridge CV','Lasso CV']
    #for c in check:
    #    idr = a.fits['name'].index(c)
    #    r = a.fits['model'][idr]
    #    r_cv = pd.DataFrame(
    #        {'alpha':a.alphas,'score':np.mean(r.cv_values_,axis=0)}
    #    )
    #    plt.plot(r_cv['alpha'],r_cv['score'])
    #    plt.xscale('log')
    #    plt.yscale('log')
    #    plt.show()
#




