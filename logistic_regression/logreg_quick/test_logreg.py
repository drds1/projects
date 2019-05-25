import numpy as np
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pylab as plt



class lrtest:


    def __init__(self):
        self.N = 1000
        self.K = 100
        self.n_targets = 1
        self.n_informative = 1



    def get_data(self):
        ''' make fake test data'''
        self.X,self.y = sklearn.datasets.make_regression(n_samples=self.N,
                                             n_features=self.K,
                                             n_informative=self.n_informative,
                                             n_targets=self.n_targets,
                                             bias=0.0,
                                             effective_rank=None,
                                             tail_strength=0.5, noise=0.0,
                                             shuffle=True, coef=False, random_state=None)


    def make_data_good_for_logreg(self):
        '''
        make the data integer and sum a set of columns
        (simulate useful and useless PADD 1 PORTS)
        ignoring others
        :return:
        '''
        minX = np.min(self.X)
        self.X = np.array(self.X-minX,dtype=int)*100
        self.y = np.sum(self.X[:,:5],axis=1,dtype=int)


    def fit_model(self):
        '''
        fit the model
        :return:
        '''
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)#LogisticRegression(random_state=0)
        self.model.fit(self.X, self.y)
        self.predictions = self.model.predict(self.X)
        #self.model.predict_proba(self.X)
        #self.model.score(self.X, self.y)
#

    def plot_model(self):
        '''
        plot model
        :return:
        '''
        plt.close()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.y,label='true')
        ax1.plot(self.predictions, label='modeled')
        plt.legend()



if __name__ == '__main__':
    x = lrtest()
    x.get_data()
    x.make_data_good_for_logreg()
    x.fit_model()
    x.plot_model()
    plt.show()
#
    #prob = x.model.predict_proba(x.X)

    #from sklearn.datasets import load_iris
    #from sklearn.linear_model import LogisticRegression
    #X, y = load_iris(return_X_y=True)
    #clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class = 'multinomial').fit(X, y)
    #clf.predict(X[:2, :])
    #clf.predict_proba(X[:2, :])
    #clf.score(X, y)



