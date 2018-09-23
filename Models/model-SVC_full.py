# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 19:56:14 2018

@author: sn06
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc = SVC(kernel='linear')

data = pd.read_csv('2018-08-17-AllCompanyQuarterly.csv')
data['Quarter end'] = pd.to_datetime(data['Quarter end'])
data['Quarter end'] = data['Quarter end'].dt.date
data = data.sort_values(by=['Company','Quarter end'])

#corr heatmap
#corr = data.corr()
#sns.heatmap(corr,cmap='plasma')

#data = data[data['Company']=='MAC']

for q in range(2002,2018):
    for w in [4,7,10]:
        if w == 10:
            ww = 1
            qq = q + 1
        else:
            ww = w + 3
            qq = q
        datatrain = data[data['Quarter end'] < date(q,w,1)]
        datatrain = datatrain.drop(['Quarter end','Company','PriceChange'],axis=1)
        for i in range(1,7):
            datatrain = datatrain.drop(['Quarter end-%s' % i],axis=1)
            datatrain = datatrain.drop(['Company-%s' % i],axis=1)
            datatrain = datatrain.drop(['PriceChange-%s' % i],axis=1)
            datatrain = datatrain.drop(['Buy/Sell-%s' % i],axis=1)
        datatrain = datatrain.drop(datatrain.columns[0],axis=1)
        datatrain = datatrain.dropna()
        
                        
        datatest = data[data['Quarter end'] <= date(q,w,1)]
        datatest = datatest[datatest['Quarter end'] >= date(q,w,1)]
        testcompany= datatest[['Quarter end','Company','PriceChange']]
        datatest = datatest.drop(['Quarter end','Company','PriceChange'],axis=1)
        for i in range(1,7):
            datatest = datatest.drop(['Quarter end-%s' % i],axis=1)
            datatest = datatest.drop(['Company-%s' % i],axis=1)
            datatest = datatest.drop(['PriceChange-%s' % i],axis=1)
            datatest = datatest.drop(['Buy/Sell-%s' % i],axis=1)
        datatest = datatest.drop(datatest.columns[0],axis=1)
        datatest = datatest.dropna()
        
    
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
        
        svc.fit(X_train,y_train)
        
        y_pred = svc.predict(X_test)
        print('%s-%s' % (q,w))
        print(accuracy_score(y_test,y_pred))
        
        
        testout = datatest.merge(testcompany,left_index=True,right_index=True)
        testout['y_pred'] = y_pred
        testout['y_test']=y_test
        testout = testout[['Quarter end','Company','Price','PriceChange','Shares split adjusted','y_test','y_pred']]
        
        adjaccuracy = testout[testout['y_pred']==1]
        print(accuracy_score(adjaccuracy['y_test'].values, adjaccuracy['y_pred'].values))
        adjaccuracy = testout[testout['y_pred']!=0]
        print(accuracy_score(adjaccuracy['y_test'].values, adjaccuracy['y_pred'].values))
        testout.to_csv('testout-%s-%s.csv' % (q,w))
