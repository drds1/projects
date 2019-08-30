import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error




# transform data class
class transform:

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1.,1.))
        self.feature_range = (-1.,1.)
        #self.X = None
        self.y = None
        self.lookback = 4


    def prepare_format(self):
        '''
        arange in to lstm format X y
        :return:
        '''
        Ny = len(y)
        ynext = np.roll(self.y,-1)
        X = np.reshape(np.tile(self.y,self.lookback+1),(self.lookback+1,Ny)).T
        for i in range(self.lookback+1):
            X[:,i] = np.roll(X[:,i],i)
        self.yn = ynext[self.lookback:]
        self.Xn = X[self.lookback:,:]


    def transform(self):
        '''
        transform the data into the lstm-required format
        :return:
        '''
        self.combined = np.hstack((self.Xn,self.yn.reshape(-1,1)))
        self.combined_t = self.scaler.fit_transform(self.combined)
        self.Xt = self.combined_t[:,:-1]
        self.yt = self.combined_t[:,-1]


    def set_model(self):
        '''
        set the lstm
        :return:
        '''
        # create and fit the LSTM network
        look_back = 1
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(1, self.lookback+1)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit_model(self):
        '''
        fit lstm
        :return:
        '''
        self.model.fit(self.Xt[:-1,:], self.yt[:-1], epochs=100, batch_size=1, verbose=2)

    def predict_next_step(self):
        '''
        evaluate one step prediction
        :return:
        '''
        return(self.model.predict(self.Xt[-1:]))


    def predict_multistep(self,nstep=10):
        '''
        predict the next nstep predictions
        :return:
        '''
        for i in range(nstep):
            newpred = self.predict_next_step()
            Xtnew = np.append([newpred],self.Xt[-1,:-1]).reshape(1,-1)
            self.Xt = np.vstack(self.Xt,Xtnew)



