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
from sklearn.metrics import accuracy_score
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
    model.add(Dropout(0.5))
    model.add(Dense(280,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(280,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(280,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(280,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

data = pd.read_csv('2018-07-15-AllCompanyQuarterly.csv')
#data = data[data['Company']!='BRK.A']
data['Quarter end'] = pd.to_datetime(data['Quarter end'])
data['Quarter end'] = data['Quarter end'].dt.date
data = data.sort_values(by=['Company','Quarter end'])

#corr heatmap
#corr = data.corr()
#sns.heatmap(corr,cmap='plasma')

#data = data[data['Company']=='MAC']

datatrain = data[data['Quarter end'] < date(2012,1,1)]
datatrain = datatrain.drop(['Quarter end','Company','PriceChange'],axis=1)
for i in range(1,7):
    datatrain = datatrain.drop(['Quarter end-%s' % i],axis=1)
    datatrain = datatrain.drop(['Company-%s' % i],axis=1)
    datatrain = datatrain.drop(['PriceChange-%s' % i],axis=1)
    datatrain = datatrain.drop(['Buy/Sell-%s' % i],axis=1)
datatrain = datatrain.drop(datatrain.columns[0],axis=1)
datatrain = datatrain.dropna()


datatest = data[data['Quarter end'] <= date(2018,1,1)]
datatest = datatest[datatest['Quarter end'] >= date(2012,1,1)]
testcompany= datatest[['Quarter end','Company','PriceChange']]
datatest = datatest.drop(['Quarter end','Company','PriceChange'],axis=1)
for i in range(1,7):
    datatest = datatest.drop(['Quarter end-%s' % i],axis=1)
    datatest = datatest.drop(['Company-%s' % i],axis=1)
    datatest = datatest.drop(['PriceChange-%s' % i],axis=1)
    datatest = datatest.drop(['Buy/Sell-%s' % i],axis=1)
datatest = datatest.drop(datatest.columns[0],axis=1)
datatest = datatest.dropna()

#datatest = datatest[datatest['Company']=='AA']
X_train = datatrain.drop(['Buy/Sell'],axis=1)
X_train = X_train.values
y_train = datatrain[['Buy/Sell']].values
y_train = y_train[:,0]

X_test = datatest.drop(['Buy/Sell'],axis=1)
X_test = X_test.values
y_test = datatest['Buy/Sell'].values

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = create_model()
le.fit_transform(y_train)
y_train = encode(y_train)
y_test = encode(y_test)


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=2)

y_pred = model.predict(X_test)
y_prob = y_pred.max(axis=-1)
y_pred = y_pred.argmax(axis=-1)
y_test = decode(y_test)

print(accuracy_score(y_test,y_pred))

testout = datatest.merge(testcompany,left_index=True,right_index=True)
testout['y_pred'] = y_pred
testout['y_test']=y_test
testout['y_prob'] = y_prob
testout = testout[['Quarter end','Company','Price','PriceChange','Shares split adjusted','y_test','y_pred','y_prob']]

adjaccuracy = testout[testout['y_pred']==1]
print(accuracy_score(adjaccuracy['y_test'].values, adjaccuracy['y_pred'].values))
adjaccuracy = testout[testout['y_pred']!=0]
print(accuracy_score(adjaccuracy['y_test'].values, adjaccuracy['y_pred'].values))
adjaccuracy = testout[testout['y_pred']==-1]
print(accuracy_score(adjaccuracy['y_test'].values, adjaccuracy['y_pred'].values))
testout.to_csv('testout-NN.csv')