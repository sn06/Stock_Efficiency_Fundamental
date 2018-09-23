# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:05:29 2018

@author: sn06
"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

endlist = []
resultlist = []
mresultlist = []
brlist = []
mbrlist = []

data = pd.read_csv('testout-LR-reg-all.csv')
data = data[data['y_pred']!=0]
data = data[data['y_pred']!=0.1]
data = data[data['y_size']<20]
data['end'] = pd.to_datetime(data['Quarter end'],format='%Y-%m-%d')
data['end'] = pd.DatetimeIndex(data['end'])
data = data.sort_values(by=['Quarter end','Company'])
data['PChangeNext']=data.groupby('Company')['PriceChange'].shift(-1)
data['result'] = ''
data['y_predabs'] = 0
data['y_predabs'][data['y_pred']>data['Price']] = 1
data['y_predabs'][data['y_pred']<data['Price']] = -1
comps = data['Company'].unique()
times = data['end'].sort_values().unique()
br = 10000
mbr = 10000

data['bs'] = data.groupby(['end','y_predabs'])['y_predabs'].transform('count')

data['reg_bs'] = data.groupby(['end'])['y_size'].transform('sum')

data['mcap'] = data['Price'] * data['Shares split adjusted']
data['mbs'] = data.groupby('end')['mcap'].transform('sum')
data['mbs'] = data['mcap'] / data['mbs']
for i in times:
    print(i)
    q = data[data['end']==i]
    q['bs'] = br * (q['y_size'] / q['reg_bs']) * q['y_predabs']
    print((q['y_size'] / q['reg_bs']).sum())
    q['result'] = q['bs']*q['PChangeNext']
    endlist.append(i)
    resultlist.append(q['result'].sum())
    br = br + q['result'].sum()+10000
    brlist.append(br)
    
    q['mbs'] = q['mbs'] * mbr
    q['mresult'] = q['mbs'] * q['PChangeNext']
    mresultlist.append(q['mresult'].sum())
    mbr = mbr + q['mresult'].sum()+10000
    mbrlist.append(mbr)

plt.plot(endlist,brlist,label='br')
plt.plot(endlist,mbrlist,label='mbr')
plt.legend()