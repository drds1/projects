import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.feature_extraction import FeatureHasher as fhe
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import keras


#keras convolutional neural network


#image width and height
nwidth = 28
nheight = 28

np.random.seed(123)  # for reproducibility



#import standard NN modules
from keras.models import Sequential


#Keras core layers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D,MaxPooling2D


#Keras CNN layers
from keras.utils import np_utils



#Load MNIST data
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#examine shape of training data set
print X_train.shape

#Plotting first sample of X_train
from matplotlib import pyplot as plt
plt.imshow(X_train[0])


#must explicitly declare a dimension for the depth of the input image. 
#For example, a full-color image with all 3 RGB channels will have a depth of 3.
#These MNIST images only have a depth of 1, but we must explicitly declare that.
#Reshape input data
X_train = X_train.reshape(X_train.shape[0], 1, nwidth, nheight)
X_test = X_test.reshape(X_test.shape[0], 1, nwidth, nheight)	

X_train = X_train.reshape(X_train.shape[0], nwidth, nheight, 1)
X_test  =  X_test.reshape(X_test.shape[0], nwidth,nheight, 1)
print X_train.shape



#final preprocessing step for the input data is to convert
#our data type to float32 and normalize our data values to the range [0, 1].
#why / 255?
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255





#examine labels
print y_train.shape
# (60000)
# Convert 1-dimensional class arrays to 10-dimensional class matrices
# this is like pandas one hot encoding e.g when classifying the numbers
# 1 - 10, we dont want a 1d array of numbers but rather a Nx10 array
# e.g if 3, class element should say [0,0,0,1,0,0,0,0,0,0] etc
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print Y_train.shape
# (60000, 10)








#!!!!!!!!!!!!!!!!!!!!!!!!!! BUILD THE CNN #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Declare Sequential model
#proven architectures from academic papers or use existing examples. See online for proven 
#CNN architectures from literature. sequential is a common choice
	
model = Sequential()

#CNN input layer	
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(nwidth,nheight,1)))
#first 3 parameters represent the number of convolution filters to use, 
#the number of rows in each convolution kernel, 
#and the number of columns in each convolution kernel, respectively.


print model.output_shape
# (None, 32, 26, 26)

# add further convolution, poo;ling and dropout layers
# need to read up online for further clarification on pooling layers
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


#Fully connected Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))





#!!!!!!!!!! COMPILE #1!!!!!!!!!!!!!!!!!!
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
              
              
              
              
              
              
              
#!!!!!!!!!!!!! fit on training data #!!!!!!!!!!!!!!!!!
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=2, verbose=1)



score = model.evaluate(X_test, Y_test, verbose=0)

#predict classifications of new images using model.predict(X_test)

op = model.predict(X_test)
#output array of (Ntest,Nclass) where each column is the probability that the image 
#belongs to that class


