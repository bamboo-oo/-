# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 15:50:03 2021

@author: 23674
"""
#清洗数据，随机抽取
from pandas import  DataFrame
import pandas as pd
import random

import numpy as np 



import csv
# 打开训练数据
train_data = 'E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random32000.csv'
dep_tit_ask_ans = pd.read_csv(open(train_data, encoding='utf8'), sep=',')
depart=dep_tit_ask_ans['department'].copy()
dep=dep_tit_ask_ans['dep'].copy()
tit=dep_tit_ask_ans['title'].copy()
ask=dep_tit_ask_ans['ask'].copy()

# 生成一个随机数列表
resultList1 = random.sample(range(1,2001),600)
resultList2 = random.sample(range(2001,4001),600)
resultList3 = random.sample(range(4001,6001),600)
resultList4 = random.sample(range(6001,8001),600)
resultList5 = random.sample(range(8001,10001),600)
resultList6 = random.sample(range(10001,12001),600)
resultList7 = random.sample(range(12001,14001),600)
resultList8 = random.sample(range(14001,16001),600)
resultList9 = random.sample(range(16001,18001),600)
resultList10 = random.sample(range(18001,20001),600)
resultList11 = random.sample(range(20001,22001),600)
resultList12 = random.sample(range(22001,24001),600)
resultList13 = random.sample(range(24001,26001),600)
resultList14 = random.sample(range(26001,28001),600)
resultList15 = random.sample(range(28001,30001),600)
resultList16 = random.sample(range(30001,32001),600)

resultList=resultList1+resultList2+resultList3+resultList4+resultList5+resultList6+resultList7+resultList8+resultList9+resultList10+resultList11+resultList12+resultList13+resultList14+resultList15+resultList16
#print(resultList)

lines_dada = pd.DataFrame(dep_tit_ask_ans.copy())
newdep=[]
newdepart=[]
newtit=[]
newask=[]
olddep=[]
olddepart=[]
oldtit=[]
oldask=[]
for i in resultList:
    newdep.append(depart[i])
    newdepart.append(dep[i])
    newtit.append(tit[i])
    newask.append(ask[i])
#    lines_dada.drop([i],axis=0)


depart=np.array(newdep)[:,np.newaxis]
dep=np.array(newdepart)[:,np.newaxis]
title=np.array(newtit)[:,np.newaxis]
askk=np.array(newask)[:,np.newaxis]
concatenate_array=np.concatenate((depart,dep,title,askk),axis=1)
data=DataFrame(concatenate_array,columns=["department","dep",'title','ask'])




df_data=dep_tit_ask_ans.drop(resultList)
#生成训练集数据
df_data.to_csv('E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random22400.csv',index=False)
#生成测试集数据
data.to_csv('E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random9600.csv',index=False)



    


