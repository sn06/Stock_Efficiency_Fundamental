# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:37:07 2018

@author: sn06
"""

import pandas as pd
from datetime import date
import datetime

x = 1
threshold = 0
yearrange = range(1994,2019)

for i in os.listdir(os.getcwd()):
    if 'csv' in i and not 'AllCompanyQuarterly' in i:
        print(i)
        if x==1:
            z = pd.read_csv(i,na_values='None')
            z['Quarter end'] = pd.to_datetime(z['Quarter end'])
            
            for j in yearrange:
                z['Quarter end'][(z['Quarter end'] > pd.to_datetime(date(int(j),11,14))) & (z['Quarter end'] <= pd.to_datetime(date(int(j)+1,2,14)))] = pd.to_datetime(date(int(j)+1,1,1))
                z['Quarter end'][(z['Quarter end'] > pd.to_datetime(date(int(j)+1,2,14))) & (z['Quarter end'] <=  pd.to_datetime(date(int(j)+1,5,14)))] = pd.to_datetime(date(int(j)+1,4,1))
                z['Quarter end'][(z['Quarter end'] >  pd.to_datetime(date(int(j)+1,5,14))) & (z['Quarter end'] <=  pd.to_datetime(date(int(j)+1,8,14)))] = pd.to_datetime(date(int(j)+1,7,1))
                z['Quarter end'][(z['Quarter end'] >  pd.to_datetime(date(int(j)+1,8,14))) & (z['Quarter end'] <=  pd.to_datetime(date(int(j)+1,11,14)))] = pd.to_datetime(date(int(j)+1,10,1))
            z['Company'] = ''
            z['Company'][z['Company']==''] = i[:i.find('_')]
            z = z.sort_values(by='Quarter end')
            z['PriceChange'] = z['Price'].pct_change()
            z['Buy/Sell'] = 0
            z['Buy/Sell'][z['PriceChange'].shift(-1) > threshold] = 1
            z['Buy/Sell'][z['PriceChange'].shift(-1) <= threshold] = -1
            z['Buy/Sell'].fillna(0)
            for j in list(z[1:]):
                z['%s-1' % j] = z[j].shift(1)
                z['%s-2' % j] = z[j].shift(2)
                z['%s-3' % j] = z[j].shift(3)
                z['%s-4' % j] = z[j].shift(4)
                z['%s-5' % j] = z[j].shift(5)
                z['%s-6' % j] = z[j].shift(6)
            x = x+1
        else:
            temp = pd.read_csv(i,na_values='None')
            temp['Quarter end'] = pd.to_datetime(temp['Quarter end'])
            for k in yearrange:
                temp['Quarter end'][(temp['Quarter end'] >  pd.to_datetime(date(int(k),11,14))) & (temp['Quarter end'] <=  pd.to_datetime(date(int(k)+1,2,14)))] = pd.to_datetime(date(int(k)+1,1,1))
                temp['Quarter end'][(temp['Quarter end'] >  pd.to_datetime(date(int(k)+1,2,14))) & (temp['Quarter end'] <=  pd.to_datetime(date(int(k)+1,5,14)))] = pd.to_datetime(date(int(k)+1,4,1))
                temp['Quarter end'][(temp['Quarter end'] >  pd.to_datetime(date(int(k)+1,5,14))) & (temp['Quarter end'] <=  pd.to_datetime(date(int(k)+1,8,14)))] = pd.to_datetime(date(int(k)+1,7,1))
                temp['Quarter end'][(temp['Quarter end'] >  pd.to_datetime(date(int(k)+1,8,14))) & (temp['Quarter end'] <=  pd.to_datetime(date(int(k)+1,11,14)))] = pd.to_datetime(date(int(k)+1,10,1))
            temp['Company'] = ''
            temp['Company'][temp['Company']==''] = i[:i.find('_')]
            temp = temp.sort_values(by='Quarter end')
            temp['PriceChange'] = temp['Price'].pct_change()
            temp['Buy/Sell'] = 0
            temp['Buy/Sell'][temp['PriceChange'].shift(-1) > threshold] = 1
            temp['Buy/Sell'][temp['PriceChange'].shift(-1) <= threshold] = -1
            temp['Buy/Sell'].fillna(0)
            for j in list(temp[1:]):
                temp['%s-1' % j] = temp[j].shift(1)
                temp['%s-2' % j] = temp[j].shift(2)
                temp['%s-3' % j] = temp[j].shift(3)
                temp['%s-4' % j] = temp[j].shift(4)
                temp['%s-5' % j] = temp[j].shift(5)
                temp['%s-6' % j] = temp[j].shift(6)
            z = z.append(temp)
z.to_csv('%s-AllCompanyQuarterly.csv' % date.today())        
del(temp)
