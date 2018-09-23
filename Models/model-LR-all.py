# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 19:56:14 2018

@author: sn06
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

le = preprocessing.LabelEncoder()

def encode(y):
    out = []
    for i in y:
        if i == 0:
            out.append([0,1,0])
        elif i == -1:
            out.append([1,0,0])
        elif i == 1:
            out.append([0,0,1])
    out = np.array(out)
    return out
    
def decode(y):
    out = []
    for i in y:
        if (i == [0,1,0]).all():
            out.append(0)
        elif (i == [1,0,0]).all():
            out.append(-1)
        elif (i == [0,0,1]).all():
            out.append(1)
    out = np.array(out)
    return out

def decode_classes(y):
    out = []
    for i in y:
        if i == 0:
            out.append(-1)
        elif i == 1:
            out.append(0)
        elif i == 2:
            out.append(1)
    out = np.array(out)
    return out    

data = pd.read_csv('2018-08-17-AllCompanyQuarterly.csv')
data['Quarter end'] = pd.to_datetime(data['Quarter end'])
data['Quarter end'] = data['Quarter end'].dt.date

data = data[data['Company']!='ASH']
data = data[data['Company']!='TMO']
data = data[data['Company']!='NTAP']
data = data[data['Company']!='RMC']
data = data[data['Company']!='THC']

data['NextPrice']=data.groupby('Company')['Price'].shift(-1)
data = data.sort_values(by=['Company','Quarter end'])
data = data.fillna(data.groupby('Company').mean())
data = data.dropna(subset=['Shares-6'])

for q in range(2002,2018):
    for w in [1,4,7,10]:
        if w==1:
            ww = 7
            qq = q-2
        if w==4:
            ww = 10
            qq = q-2
        if w==7:
            ww = 1
            qq = q-1
        if w==10:
            ww = 4
            qq = q-1
        
        datatrain = data[data['Quarter end'] < date(q,w,1)]
        #datatrain = datatrain[datatrain['Quarter end'] >= date(qq,ww,1)]
        datatrain = datatrain.drop(['Quarter end','Company','PriceChange','Shares'],axis=1)
        for i in range(1,7):
            datatrain = datatrain[datatrain.columns.drop(list(datatrain.filter(regex='-%s' % i)))]
        datatrain = datatrain.drop(datatrain.columns[0],axis=1)
        datatrain = datatrain.dropna()
        
        datatest = data[data['Quarter end'] <= date(q,w,1)]
        datatest = datatest[datatest['Quarter end'] >= date(q,w,1)]
        testcompany= datatest[['Quarter end','Company','PriceChange']]
        datatest = datatest.drop(['Quarter end','Company','PriceChange','Shares'],axis=1)
        for i in range(1,7):
            datatest = datatest[datatest.columns.drop(list(datatest.filter(regex='-%s' % i)))]
        datatest = datatest.drop(datatest.columns[0],axis=1)
        datatest = datatest.dropna()
        
        X_train = datatrain.drop(['Buy/Sell','NextPrice'],axis=1)
        X_train = X_train.values
        y_train = datatrain[['NextPrice']].values
        y_train = y_train[:,0]
        
        X_test = datatest.drop(['Buy/Sell','NextPrice'],axis=1)
        X_test = X_test.values
        y_test = datatest['NextPrice'].values
        
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LinearRegression(normalize=True)
        
        history = model.fit(X_train,y_train)
        
        y_pred = model.predict(X_test)

        testout = datatest.merge(testcompany,left_index=True,right_index=True)
        testout['y_pred'] = y_pred
        
        testout['y_test']=y_test
        testout['y_test'][testout['y_test']==0] = 0.1
        testout['y_pred'][testout['y_pred']<0] = 0.1
        testout['y_size'] = testout['y_pred'] / testout['Price']
        testout['y_size'][testout['y_pred']<testout['Price']] = testout['Price'] / testout['y_pred']
        
        print('%s-%s' % (q,w))
        print(r2_score(testout['y_test'],testout['y_pred']))
        plt.scatter(testout['y_test'].values,testout['y_pred'].values)
        plt.show()
        testout = testout[['Quarter end','Company','Price','PriceChange','Shares split adjusted','y_test','y_pred','y_size']]

        testwin = testout.copy()
        testwin['y_pred']=testwin['y_test']
        testwin['y_size'] = testwin['y_pred'] / testwin['Price']
        testwin['y_size'][testwin['y_pred']<testwin['Price']] = testwin['Price'] / testwin['y_pred']
        testout.to_csv('testout-LR-reg-%s-%s.csv' % (q,w))
        testwin.to_csv('testwin-LRWIN-reg-%s-%s.csv' % (q,w))