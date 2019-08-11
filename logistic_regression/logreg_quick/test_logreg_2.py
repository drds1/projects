from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import numpy as np

class log_reg_heartdisease:
    '''
    use logistic regression analysis to predict
    the likelihood of heart disease
    '''


    def __init__(self):
        self.file = './framingham.csv'
        self.frac_train = 0.75




    def load_data(self,verbose = True):
        self.df = pd.read_csv(self.file).dropna().sample(frac=1)
        self.y = self.df['TenYearCHD']
        self.X = self.df.drop(columns=['TenYearCHD'])
        n = len(self.y)
        idxtrain = np.arange(int(self.frac_train*n))
        idxtest = np.delete(np.arange(n),idxtrain)
        self.Xtrain = np.array(self.X.iloc[idxtrain,:])
        self.ytrain = np.array(self.y.iloc[idxtrain])
        self.Xtest  = np.array(self.X.iloc[idxtest,:])
        self.ytest = np.array(self.y.iloc[idxtest])
        if verbose is True:
            print('Loading data...')
            np.where(self.y == 1)
            print(self.df.groupby('TenYearCHD').count())



    def fit(self,X,y):
        self.clf = LogisticRegressionCV(random_state=0, solver='lbfgs',
                                 multi_class='multinomial').fit(X, y)




    def predict(self,X,threshold = 0.5,verbose = True):
        classification = np.zeros(len(X))
        probs = self.clf.predict_proba(X)
        classification[probs[:,1] >= threshold] = 1
        if verbose is True:
            print('predicting classifications...')
            positive = np.where(classification == 1)[0]
            negative = np.where(classification == 0)[0]
            print('found ',len(positive),' positive and ',len(negative),' negative')
        return(classification)




if __name__ == '__main__':
    x = log_reg_heartdisease()
    x.load_data()
    x.fit(x.Xtrain,x.ytrain)
    df = x.df
    predictions = x.predict(x.Xtrain,threshold=0.5)



