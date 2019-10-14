from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from utils import *
from keras.models import Sequential
from keras import layers


class sentence_classifier:

    def __init__(self):
        self.y = None
        self.sentences = None
        self.sentences_train = None
        self.y_train = None
        self.sentences_test = None
        self.y_test = None
        self.num_words = 5000
        self.maxlen = 100


    def split_train_test(self,test_size = 0.25,random_state=1000):
        '''
        split the sentence data into train and test samples
        :return:
        '''
        # split into train test data
        self.sentences_train, self.sentences_test, self.y_train, self.y_test = train_test_split(
            sentences, y,
            test_size=test_size,
            random_state=random_state)

    def convert_to_numeric(self):
        '''
        convert the sentence data into numeric form
        :return:
        '''
        tokenizer = Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(self.sentences_train)
        X_train = tokenizer.texts_to_sequences(self.sentences_train)
        X_test = tokenizer.texts_to_sequences(self.sentences_test)  # Adding 1 because of  reserved 0 index
        self.vocab_size = len(tokenizer.word_index) + 1
        maxlen = self.maxlen
        self.X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        self.X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
        self.tokenizer = tokenizer


    def set_default_model(self,embedding_dim=100):
        '''
        specify default model
        :return:
        '''
        model = Sequential()
        model.add(layers.Embedding(self.vocab_size, embedding_dim, input_length=self.maxlen))
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def fit(self):
        '''
        :return:
        '''
        self.history = self.model.fit(self.X_train, self.y_train,
                            epochs=10,
                            validation_data=(self.X_test, self.y_test),
                            batch_size=10)


    def predict(self,sentences):
        '''
        enter a list of sentence string and predict the label probability
        :param sentence:
        :return:
        '''
        tokenizer = self.tokenizer
        X = tokenizer.texts_to_sequences(sentences)
        X = pad_sequences(X, padding='post', maxlen=self.maxlen)
        return self.model.predict(X)[:,0]


    def predict_class(self,sentences):
        '''
        enter a list of sentence string and predict the label class
        :param sentence:
        :return:
        '''
        tokenizer = self.tokenizer
        X = tokenizer.texts_to_sequences(sentences)
        X = pad_sequences(X, padding='post', maxlen=self.maxlen)
        return self.model.predict_classes(X)[:,0]


if __name__ == '__main__':




    #load input sentence data
    df = load_sentiment_data(path = './input/sentiment labelled sentences',
                            file_dict={'yelp': 'yelp_labelled.txt',
                                       'amazon': 'amazon_cells_labelled.txt',
                                       'imdb': 'imdb_labelled.txt'})

    #isolate only on the yelp data
    df_yelp = df[df['source'] == 'yelp']
    sentences = df_yelp['sentence'].values
    y = df_yelp['label'].values


    #instantiate nlp model
    nlp = sentence_classifier()
    nlp.sentences = sentences
    nlp.y = y

    nlp.split_train_test()
    nlp.convert_to_numeric()
    nlp.set_default_model()
    nlp.fit()
    predictions_prob = nlp.predict(sentences[:])
    predictions_lab = nlp.predict_class(sentences[:])















