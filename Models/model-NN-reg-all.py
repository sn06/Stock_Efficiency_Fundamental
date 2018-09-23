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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

le = preprocessing.LabelEncoder()

adm = Adam(lr=0.0001)

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

def create_model():
    model = Sequential()
    model.add(Dense(280,input_dim=X_train.shape[1],activation='relu'))
    model.add(Dense(280,activation='relu'))
    model.add(Dense(280,activation='relu'))
    model.add(Dense(280,activation='relu'))
    model.add(Dense(280,activation='relu'))
    model.add(Dense(1,activation='relu'))
    model.compile(optimizer=adm, loss='mean_squared_error', metrics=['mae'])
    return model

def remove_anomalies():
    finaldata = pd.DataFrame(columns=data.columns)
    for j in data['Company'].unique():
        data_comp = data[data['Company']==j]
        print(j)
        for k in data['Quarter end'].unique():
            data_comp = data_comp[data_comp['Quarter end']==k]
            for i in list(data_comp):
                if i=='PriceChange':
                    break
                if data_comp[i].dtype!='object':
                    if data_comp[i].dtype!='int64':
                        quantile_val = data_comp[i].dropna().quantile(0.999)
                        if quantile_val > 0:
                            data_comp = data_comp[data_comp[i] < quantile_val]
            finaldata = finaldata.append(data_comp)
    finaldata.to_csv('finaldata.csv')
    data = finaldata.copy()
    del(finaldata)

data = pd.read_csv('2018-08-17-AllCompanyQuarterly.csv')
data['Quarter end'] = pd.to_datetime(data['Quarter end'])
data['Quarter end'] = data['Quarter end'].dt.date

data = data[data['Company']!='BRK.A']
data = data[data['Company']!='RRD']

data['NextPrice']=data.groupby('Company')['Price'].shift(-1)
data = data.sort_values(by=['Company','Quarter end'])

data = data.fillna(data.interpolate())
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
            datatrain = datatrain.drop(['Quarter end-%s' % i],axis=1)
            datatrain = datatrain.drop(['Company-%s' % i],axis=1)
            datatrain = datatrain.drop(['PriceChange-%s' % i],axis=1)
            datatrain = datatrain.drop(['Buy/Sell-%s' % i],axis=1)
            datatrain = datatrain.drop(['Shares-%s' % i],axis=1)
        datatrain = datatrain.drop(datatrain.columns[0],axis=1)
        datatrain = datatrain.dropna()
        
        datatest = data[data['Quarter end'] <= date(q,w,1)]
        datatest = datatest[datatest['Quarter end'] >= date(q,w,1)]
        testcompany= datatest[['Quarter end','Company','PriceChange']]
        datatest = datatest.drop(['Quarter end','Company','PriceChange','Shares'],axis=1)
        for i in range(1,7):
            datatest = datatest.drop(['Quarter end-%s' % i],axis=1)
            datatest = datatest.drop(['Company-%s' % i],axis=1)
            datatest = datatest.drop(['PriceChange-%s' % i],axis=1)
            datatest = datatest.drop(['Buy/Sell-%s' % i],axis=1)
            datatest = datatest.drop(['Shares-%s' % i],axis=1)
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
        
        model = create_model()
        
        history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=60,verbose=0)
        
        y_pred = model.predict(X_test)

        testout = datatest.merge(testcompany,left_index=True,right_index=True)
        testout['y_pred'] = y_pred
        
        testout['y_test']=y_test
        testout['y_test'][testout['y_test']==0] = 0.1
        testout['y_size'] = testout['y_pred'] / testout['Price']
        testout['y_size'][testout['y_pred']<testout['Price']] = testout['Price'] / testout['y_pred']
        testout = testout[testout['y_pred']<100]
        
        print('%s-%s' % (q,w))
        print(r2_score(testout['y_test'],testout['y_pred']))
        plt.plot(history.history['mean_absolute_error'],label='mae')
        plt.plot(history.history['val_mean_absolute_error'],label='v_mae')
        plt.legend()
        plt.show()
        plt.scatter(testout['y_test'].values,testout['y_pred'].values)
        plt.show()
        testout = testout[['Quarter end','Company','Price','PriceChange','Shares split adjusted','y_test','y_pred','y_size']]

        testwin = testout.copy()
        testwin['y_pred']=testwin['y_test']
        testwin['y_size'] = testwin['y_pred'] / testwin['Price']
        testwin['y_size'][testwin['y_pred']<testwin['Price']] = testwin['Price'] / testwin['y_pred']
        testout.to_csv('testout-NN-reg-%s-%s.csv' % (q,w))
        testwin.to_csv('testwin-WIN-reg-%s-%s.csv' % (q,w))