# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:39:06 2019

"""

import pandas as pd
import numpy as mp
from nsepy import get_history
from datetime import date
#all nifty 50
dict1 = ['BPCL','M&M','HDFC','IOC','MARUTI','HDFC','TITAN','HINDUNILVR','EICHERMOT',
        'KOTAKBANK','ASIANPAINT','DRREDDY', 'SUNPHARMA', 'NTPC','LT', 'BAJAJFINSV','BRITANNIA',
        'BAJAJ-AUTO', 'HINDALCO', 'HEROMOTOCO', 'GAIL', 'POWERGRID', 'AXISBANK', 'INFY',
        'TECHM', 'BAJFINANCE', 'NESTLEIND', 'ITC', 'UPL', 'HCLTECH', 'VEDL', 'TCS', 
         'ICICIBANK', 'TATAMOTORS', 'RELIANCE', 'ADANIPORTS', 'WIPRO','ONGC', 'TATASTEEL',
         'ULTRACEMCO', 'COALINDIA', 'JSWSTEEL', 'INFRATEL', 'CIPLA','BHARTIARTL','GRASIM',
         'SBIN', 'INDUSINDBK','ZEEL','YESBANK']

#i = 'BPCL'

#for stock in dict1:
 #   stock = '%s' % stock
 #   stock= get_history(symbol=(stock),start=date(2018,1,1),end=date(2019,1,1))
 #   print(stock.head())
#the below method works
 #download data in dict2
dict2={}    
for d in dict1:
      dict2[d] = get_history(d,start=date(2018,1,1), end=date(2019,1,1))     
      dict2[d]=(dict2[d].iloc[:,3:13])



nifty = get_history(symbol='NIFTY', start=date(2018,1,1), end=date(2019,1,1),index=True)    
print(nifty.head(247))

'''i=30
nifty.iloc[i]['Close']

for i in range(0,247,30):
    if i+30>247:
        break
    else:
        pct_change=(nifty.iloc[i+30]['Close']-nifty.iloc[i]['Close'])*100/nifty.iloc[i+30]['Close']
        print(pct_change)
'''

#differentiate  volatile and non volatile.
volatile=[]
nonvolatile=[]
n=0
v=0

for d in dict2:
     for i in range(0,247,20):
        if i+20>247:
            break
        else:
           # print(d)
            pchange=(dict2[d].iloc[i+20]['Close']-dict2[d].iloc[i]['Close'])*100/dict2[d].iloc[i]['Close']
            pct_change=(nifty.iloc[i+20]['Close']-nifty.iloc[i]['Close'])*100/nifty.iloc[i]['Close']
            #print(pct_change)
            if abs(pct_change) > 0.65*abs(pchange):
                n=n+1
            else:
                v=v+1
     if n>v:
        nonvolatile.append(d)
     else:
        volatile.append(d)


#download volatile 
vtile={}
for v in volatile:
      vtile[v] = get_history(v,start=date(2018,1,1), end=date(2019,1,1))     
      vtile[v]=(vtile[v].iloc[:,3:13])

#download non volatile.
nvtile={}
for nv in nonvolatile:
      nvtile[nv] = get_history(nv,start=date(2018,1,1), end=date(2019,1,1))     
      nvtile[nv]=(nvtile[nv].iloc[:,3:13])

#non volatile mf



import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import mpl_finance
import matplotlib
import pylab
matplotlib.rcParams.update({'font.size': 9})      





#function  for relative strength index
def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array


def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=50, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow




vlist=[]
for v in vtile:
    rsi = rsiFunc(vtile[v].iloc[:]['Close'])
    emaslow, emafast, macd = computeMACD(vtile[v].iloc[:]['Close'])
    if emafast[-1] > emaslow[-1]:
        macdrating = 8
    else:
        macdrating = 0

    print(rsi[-1])
    if rsi[-1] < 25:
        rsirating = 8
    if 25 < rsi[-1] < 60:
        rsirating = 4
    else:
        rsirating = 0
        print("Stock over-bought")
        
    technicalrating = macdrating + rsirating
    print(technicalrating)    
    if technicalrating == 12:
        vlist.append(v)
    
    
nlist=[]    
for nv in nvtile:
    rsi = rsiFunc(nvtile[nv].iloc[:]['Close'])
    emaslow, emafast, macd = computeMACD(nvtile[nv].iloc[:]['Close'])
    if emafast[-1] > emaslow[-1]:
        macdrating = 8
    else:
        macdrating = 0

    print(rsi[-1])
    if rsi[-1] < 25:
        rsirating = 8
    if 25 < rsi[-1] < 60:
        rsirating = 4
    else:
        rsirating = 0
        print("Stock over-bought")
        
    technicalrating = macdrating + rsirating
    print(technicalrating)        
        
    if technicalrating == 12:
        nlist.append(nv)
        




    
