from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

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


    def get_confusion_matrix(self,threshold,Xtest,ytest):
        '''
        return the confusion matrix for the threshold and test data
        :param threshold:
        :param Xtest:
        :param ytest:
        :return:
        '''
        predictions = self.predict(Xtest,threshold=threshold)
        cm = confusion_matrix(ytest,predictions)
        return(cm)



    def ROC_analysis(self,X,y,threshold = np.logspace(-2,0,30)):
        '''
        Perform the ROC analysis for the selected input threhold values
        :param X:
        :param y:
        :param threshold:
        :return:
        '''
        true_positive_fraction = []
        false_positive_fraction = []
        for t in threshold:
            cm = self.get_confusion_matrix(t,X,y)
            tp = cm[1,1]/np.sum(cm[1,:])
            fp = cm[0,1]/np.sum(cm[0,:])
            true_positive_fraction.append(tp)
            false_positive_fraction.append(fp)

        #save the ROC info
        df_op = pd.DataFrame({'threshold':threshold,
                               'true positive fraction':true_positive_fraction,
                               'false positive fraction':false_positive_fraction})
        df_op.sort_values(by='false positive fraction',inplace=True)

        #compute the integral
        x = df_op[['false positive fraction']].values
        y = df_op[['true positive fraction']].values
        dx = x[1:] - x[:-1]
        self.AOC = np.sum(y[1:]*dx)
        self.ROC_output = df_op


    def plot_ROC_analysis(self,title = 'ROC Logistic Regression'):
        '''
        Plot the Results of the ROC analysis
        :return:
        '''
        plt.close()
        tp = self.ROC_output['true positive fraction']
        fp = self.ROC_output['false positive fraction']
        #idsort = np.argsort(fp)
        #fp = fp[idsort]
        #tp = tp[idsort]
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(fp,tp,marker='o',label='AOC = '+np.str(np.round(self.AOC,2)))
        ax1.set_xlabel('False Positive Fraction\n(1 - Specificity)')
        ax1.set_ylabel('True Positive Fraction\n(Sensistivity)')
        ax1.set_xlim([0,1])
        ax1.set_ylim([0,1])
        ax1.plot([0,1],[0,1],ls='--',label=None)
        ax1.set_title(title)
        plt.tight_layout()
        plt.legend()




if __name__ == '__main__':
    x = log_reg_heartdisease()
    x.load_data()
    x.fit(x.Xtrain,x.ytrain)
    df = x.df
    predictions = x.predict(x.Xtrain,threshold=0.5)
    x.ROC_analysis(x.Xtest,x.ytest,threshold=np.logspace(-2,0,30))
    dfroc = x.ROC_output
    x.plot_ROC_analysis()
    plt.savefig('ROC_analysis.pdf')




