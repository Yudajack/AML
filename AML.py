# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 08:56:23 2021

@author: gaoyu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:07:34 2021

@author: gaoyu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:53:51 2021

@author: gaoyu
"""

import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import ast
import json,urllib.request
from scipy.stats import norm
import copy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math
from varclushi import VarClusHi


def IV(target_df,var):
    IV_v=0
    count_list=target_df['rating'].groupby(target_df[var]).count()
    default_list=target_df['rating'].groupby(target_df[var]).sum()
    for i in range(len(count_list)):
        f_g=(count_list.iloc[i]-default_list.iloc[i])/N_GT
        f_b=default_list.iloc[i]/N_BT
        IV_i=(f_g-f_b)*np.log(f_g/f_b)
        IV_v=IV_v+IV_i
    return IV_v

def bining(df, col_name):
    category_col=[]
    for i in range(len(df)):
        if df[col_name][i]<= np.percentile(df[col_name],20):
            category_col.append("1")
        elif df[col_name][i]<= np.percentile(df[col_name],40):
            category_col.append("2")
        elif df[col_name][i]<= np.percentile(df[col_name],60):
            category_col.append("3")
        elif df[col_name][i]<= np.percentile(df[col_name],80): 
            category_col.append("4")
        else:
            category_col.append("5")
    #Every ratio has a category
    return category_col
 
filename = r"C:\Users\gaoyu\Desktop\AML and ATF Modelling Assignment data_std.xlsx" 
df = pd.read_excel(filename)

#随机取70%data作为training set，剩下30%作为testset
trainingset_index=np.random.choice(range(len(df)),size=math.floor(0.7*len(df)),replace=False)
trainingset=copy.deepcopy(df.loc[trainingset_index,:])
testset_index = [item for item in df.index if item not in trainingset_index]
testset=copy.deepcopy(df.loc[testset_index,:])

trainingset.index=range(0,len(trainingset))
testset.index=range(0,len(testset))

#记录training set，testset的客户ID
training_list=list(trainingset['cust_id_masked'])
test_list=list(testset['cust_id_masked'])

trainingset.drop('cust_id_masked',axis=1,inplace=True)
testset.drop('cust_id_masked',axis=1,inplace=True)

GRP_list=pd.DataFrame()

#bining分组
for i in list(trainingset.columns)[:-1]:
    GRP_list[i]=bining(trainingset, i)
GRP_list['rating']=copy.deepcopy(trainingset['rating'])    

#计算IV值
N_BT=np.sum(trainingset['rating']) #900
N_T=len(trainingset['rating']) #9012
N_GT=N_T-N_BT  #8112
IV_list={}

for i in list(GRP_list.columns)[:-1]:
    target_df=GRP_list[[i,'rating']]
    IV_list[i]=IV(target_df,i)

IV_list_Series=pd.Series(IV_list)
IV_list_Series2=IV_list_Series[IV_list_Series!=np.inf]
IV_list_Series_drop=IV_list_Series[IV_list_Series<=0.3]
drop_index=list(IV_list_Series_drop.index)
get_index=list(IV_list_Series2[IV_list_Series2>0.3].index)

#clustering，计算所取的变量
trainingset_afterIV=trainingset[get_index]

df_c=copy.deepcopy(trainingset_afterIV)

model=VarClusHi(df_c,maxeigval2=.7,maxclus=None)
model.varclus()
rsquare_table=model.rsquare

targetvar_list=[]
for name,group in rsquare_table.groupby('Cluster'):
        #print(group.index)
        targetvar_list.append(group['Variable'].iloc[np.argmin(group['RS_Ratio'])])     
#所取变量记录在targetvar_list中


