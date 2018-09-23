# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:05:29 2018

@author: sn06
"""

import pandas as pd
import matplotlib.pyplot as plt

endlist = []
resultlist = []
mresultlist = []
brlist = []
mbrlist = []

data = pd.read_csv('testout-all.csv',usecols=[1,2,3,4,5,6,7])
data['end'] = pd.to_datetime(data['Quarter end'],format='%Y-%m-%d')
data['PChangeNext'] = data['PriceChange'].shift(-1)

data['result'] = ''
comps = data['Company'].unique()
times = data['end'].sort_values().unique()
br = 10000
mbr = 10000

data2 = pd.read_csv('testout-NN-reg-all.csv',usecols=[0,1,2,3,4,7])
data = data.merge(data2,how='left',on=['Quarter end','Company','Price','PriceChange','Shares split adjusted'])

data['bs'] = data.groupby(['end','y_pred'])['y_pred'].transform('count')
data['bs'][data['y_pred']!=1] = 0

data['reg_bs'] = data.groupby(['end'])['y_size'].transform('sum')
data['reg_bs'][data['y_pred']!=1] = 0

data['mcap'] = data['Price'] * data['Shares split adjusted']
data['mbs'] = data.groupby('end')['mcap'].transform('sum')
data['mbs'] = data['mcap'] / data['mbs']
for i in times:
    print(i)
    q = data[data['end']==i]
    q['bs'][q['bs']!=0] = br * (q['y_size'] / q['reg_bs'])
    q['result'] = q['bs']*q['PChangeNext']
    endlist.append(i)
    resultlist.append(q['result'].sum())
    br = br + q['result'].sum()
    brlist.append(br)
    
    q['mbs'] = q['mbs'] * mbr
    q['mresult'] = q['mbs'] * q['PChangeNext']
    mresultlist.append(q['mresult'].sum())
    mbr = mbr + q['mresult'].sum()
    mbrlist.append(mbr)

plt.plot(endlist,brlist,label='br')
plt.plot(endlist,mbrlist,label='mbr')
plt.legend()