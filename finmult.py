# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:35:54 2019

@author:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nsepy import get_history
from datetime import date
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
from keras.models import Model
#neurons with op 0 ar deactivated

dataset = get_history(symbol='HDFCBANK',
                   start=date(2002,1,1),
                   end=date(2019,1,1))

dataset =  dataset.replace(np.nan,0)
training = dataset.iloc[:,4:13].values


                
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training[:,0:9] = sc.fit_transform(training[:,:])

timestep = 90
x = []
y=[]

for i in  range(len(dataset)-timestep-1):
    t = []
    for j in range(0,timestep):
        t.append(training[[(i+j)], :])
        
    x.append(t)
    y.append(training[i+timestep,1])    
    

x,y = np.array(x), np.array(y)
 
x = np.reshape(x,( x.shape[0],timestep,9))

print(x.shape)


regressor = Sequential()

#first layer 
#regressor.add(act)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape= (x.shape[1],9)))
regressor.add(Dropout(0.2))

#second layer 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#third layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#temp fifth layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

 
#fourth layer
regressor.add(LSTM(units = 64, return_sequences = False))
regressor.add(Dropout(0.2))


#output layer with just one output that is the next price
regressor.add(Dense(units = 1))#prediction of the next day

regressor.compile(optimizer = 'adam', loss='mean_squared_error')
#finds golabl minima efficiently

#fitting rnn to training set
regressor.fit(x, y, epochs=1, batch_size= 32) 

regressor.summary()

dataset2 = get_history(symbol='HDFCBANK',
                   start=date(2019,1,2),
                   end=date(2019,4,1))
#test dataset
testing = dataset2.iloc[:,4:13].values
dataset_total =  get_history(symbol='HDFCBANK',
                   start=date(2002,1,1),
                   end=date(2019,4,1))
inputs = dataset_total[len(dataset_total) - len(dataset2) - timestep:].values























   