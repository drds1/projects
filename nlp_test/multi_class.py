#text multi classification problem
#https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns

def process_text(text_features):
    '''
    convert text to numeric form
    :param text_features:
    :return:
    '''
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(text_features)
    tfidf_transformer = TfidfTransformer()
    text_numeric = tfidf_transformer.fit_transform(X_train_counts)
    return text_numeric




class text_multi_class:
    '''
    general tet classifier to convert declared to actual destinations
    '''

    def __init__(self):
        self.text_process_fn = process_text

    def fit_model(self,sentences,labels,model = MultinomialNB()):
        '''
        convert data frame into train and test samples
        :return:
        '''
        X_train, X_test, y_train, y_test = train_test_split(sentences, labels,
                                                            random_state=0)
        X_train_tfidf = self.text_process_fn(X_train)
        clf = model.fit(X_train_tfidf, y_train)
        return clf

    def multi_model_test(self,sentences, labels, models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0)],
    CV = 5):
        '''
        models: list of model classifiers
        CV = 5 CV fold cross validation
        :return:
        '''
        features = self.text_process_fn(sentences)
        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        for model in models:
          model_name = model.__class__.__name__
          accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
          for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
        self.cv_df = cv_df

    def plot_multi_model_test(self,plot_file = 'classifier_score.pdf'):
        '''
        analyse results of multi model testing
        :return:
        '''
        sns.boxplot(x='model_name', y='accuracy', data=self.cv_df)
        sns.stripplot(x='model_name', y='accuracy', data=self.cv_df,
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
        plt.savefig(plot_file)
        plt.close()



if __name__ == '__main__':
    # load data
    df = pd.read_csv('./input/cc.csv').iloc[:,[5,1]]
    df.columns = ['text','label']
    df = df.dropna()
    df = df.sample(n=5000, random_state=1534)
    sentences,labels = df.values[:,0],df.values[:,1]

    clf = text_multi_class()
    clf.multi_model_test(sentences,labels)
    clf.plot_multi_model_test(plot_file='class_test.pdf')











