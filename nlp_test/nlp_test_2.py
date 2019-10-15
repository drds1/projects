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



class demo_multi_class:


    def __init__(self):
        self.df = None
        self.resample = False
        self.random_state = 1534
        self.features = None
        self.labels = None


    def prepare_df(self):
        '''
        convert labels to ints and prepare feature/label columns
        :return:
        '''
        df = self.df
        # clean and convert labels to indicees
        df.columns = ['ccn', 'text_lab']
        df.dropna(subset=['ccn'], inplace=True)
        df['label'] = pd.Categorical(df['text_lab'])
        df['label'] = df['label'].cat.codes

        df['category_id'] = df['text_lab'].factorize()[0]
        category_id_df = df[['text_lab', 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[['category_id', 'text_lab']].values)
        df.head()

        # random sample data frame
        if self.resample is True:
            df = df.sample(n=5000, random_state=self.random_state)
        self.df = df

        # convert to feature format
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                                     encoding='latin-1', ngram_range=(1, 2),
                                     stop_words='english')
        self.features = tfidf.fit_transform(df.ccn).toarray()
        self.labels = df.label
        print(features.shape)


    def summarise_sample_labels(self, plot_file='sample_distribution.pdf'):
        '''
        generate a figure summarising the sample stats
        :param plot_file:
        :return:
        '''
        # number of entries in each class
        df = self.df.copy()
        fig = plt.figure(figsize=(8, 6))
        df.groupby('text_lab').count().iloc[:, 0].plot.bar(ylim=0)
        plt.savefig(plot_file)
        plt.close()


    def feature_selection_analysis(self):
        '''
        isolate the most correlated words with each class
        :return:
        '''
        features = self.features
        labels = self.labels
        tfidf = self.tfidf
        category_to_id = self.category_to_id
        # feature selection analysis
        N = 2
        for Product, category_id in sorted(category_to_id.items()):
            features_chi2 = chi2(features, labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("# '{}':".format(Product))
            print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
            print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

