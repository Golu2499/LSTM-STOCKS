# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 09:23:20 2019

@author: GanduGolu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nsepy import get_history
from datetime import date

dataset = get_history(symbol='HDFCBANK',
                   start=date(2002,1,1),
                   end=date(2019,1,1))
                   
newset = get_history(symbol='HDFCBANK', start=date(2002,1,1), end=date(2019,1,1))                   

training = dataset.iloc[:,3:4].values
training2 = dataset.iloc[:,4:13].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training = sc.fit_transform(training)

#creating timestamp 
xtrain = []
ytrain = []
for i in range(60,len(training)):
    xtrain.append(training[i-60:i,0])
    ytrain.append(training[i,0])
    
xtrain, ytrain = np.array(xtrain), np.array(ytrain)
#reshape data first param reshape, shape[0] first param, shape[10] gets timestamp
xtrain = np.reshape(xtrain,( xtrain.shape[0],xtrain.shape[1],1))
# this form of input is rewuired by keras

#ytrain = np.reshape(ytrain,( ytrain.shape[0],ytrain.shape[1],1))



import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
from keras.models import Model
#neurons with op 0 ar deactivated
from keras.layers.advanced_activations import PReLU
#RNN init
regressor = Sequential()

#first layer 
#regressor.add(act)
regressor.add(LSTM(units = 512, return_sequences = True, input_shape= (xtrain.shape[1],1)))
regressor.add(Dropout(0.2))

#second layer 
regressor.add(LSTM(units = 512, return_sequences = True))
regressor.add(Dropout(0.2))

#third layer
regressor.add(LSTM(units = 512, return_sequences = True))
regressor.add(Dropout(0.2))
#temp fifth layer
regressor.add(LSTM(units = 512, return_sequences = True))
regressor.add(Dropout(0.2))

 
#fourth layer
regressor.add(LSTM(units = 64, return_sequences = False))
regressor.add(Dropout(0.2))


#output layer with just one output that is the next price
regressor.add(Dense(units = 1))#prediction of the next day

regressor.compile(optimizer = 'adam', loss='mean_squared_error')
#finds golabl minima efficiently

#fitting rnn to training set
regressor.fit(xtrain, ytrain, epochs=1, batch_size= 128) 
    
#model is readu

dataset2 = get_history(symbol='HDFCBANK',
                   start=date(2019,1,2),
                   end=date(2019,4,1))
#test dataset
testing = dataset2.iloc[:,3:4].values

#dataset3 = get_history(symbol='HDFCBANK', start=date(2015,1,1), end=date(2019,4,1))

#we have to add the last sixty values in dataset  1 into dataset 2 
#dataset = dataset3.iloc[:,3:4].values
    
dataset_total = pd.concat((dataset['Open'], dataset2['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset2) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 122):
    #X_test = []
    X_test.append(inputs[i-60:i, 0])    


X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results


plt.plot(testing, color = 'red', label = 'HDFCBANK Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title(' 4 Month Predicted Movement')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()




