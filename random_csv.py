# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 15:50:03 2021

@author: 23674
"""

from pandas import  DataFrame
import pandas as pd
import random
from random import randint
import numpy as np 

'''
oldf='E:\代码及数据集\Chinese-medical-dialogue-data-master\Chinese-medical-dialogue-data-master\外科\外科.csv'
dep_tit_ask_ans = pd.read_csv(open(oldf, encoding='utf8'), sep=',',error_bad_lines=False)
depart=dep_tit_ask_ans['department'].copy()
tit=dep_tit_ask_ans['title'].copy()
ask=dep_tit_ask_ans['ask'].copy()
#newf=open('E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/儿科/耳鼻喉科/random8000.csv','w',encoding='UTF-8')
depart=dep_tit_ask_ans["department"].copy()
n = 0
randep=[]
randepart=[]
rantit=[]
ranask=[]
for i in range(len(depart)):
   if depart[i] == '神经脑外科':
    n+=1
    randepart.append('外科')
    randep.append(depart[i])
    rantit.append(tit[i])
    ranask.append(ask[i])
    



# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
resultList = random.sample(range(0,n),2000)

newdep=[]
newdepart=[]
newtit=[]
newask=[]
for i in resultList:
    newdep.append(randepart[i])
    newdepart.append(randep[i])
    newtit.append(rantit[i])
    newask.append(ranask[i])


depart=np.array(newdep)[:,np.newaxis]
dep=np.array(newdepart)[:,np.newaxis]
title=np.array(newtit)[:,np.newaxis]
askk=np.array(newask)[:,np.newaxis]
#print("frequence_array.shape",frequence_array.shape)
concatenate_array=np.concatenate((depart,dep,title,askk),axis=1)
#print("concatenate_array",concatenate_array)
#print("concatenate_array",concatenate_array.shape)
data=DataFrame(concatenate_array,columns=["department","dep",'title','ask'])




    
#oldf.close()
data.to_csv('E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/外科/神经脑外科/random2000_1.csv',index=False)


'''

import csv
# 打开训练数据
train_data = 'E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random32000.csv'
dep_tit_ask_ans = pd.read_csv(open(train_data, encoding='utf8'), sep=',')
depart=dep_tit_ask_ans['department'].copy()
dep=dep_tit_ask_ans['dep'].copy()
tit=dep_tit_ask_ans['title'].copy()
ask=dep_tit_ask_ans['ask'].copy()
# 打开训练标签
#train_label = open('train_label.csv','r',encoding='UTF-8')
#validation_label = open('validation_label.csv','w',encoding='UTF-8')

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
#data_array = np.array(lines_dada)
# 然后转化为list形式
#data_list =data_array.tolist()
#lines_dada = csv.reader(train_data) 
#lines_data = [row for row in lines_dada]
#lines_label = train_label.readlines()
#df=pd.DataFrame(columns=("department","dep",'title','ask'))
#for i in resultList:
#df.ix[1:9600,0:3]=lines_dada[resultList]/["department","dep",'title','ask']
#data_list.drop([resultList],axis=0)
#validation_data.write(data_list[i])
#    validation_label.write(lines_label[i])
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
#print("frequence_array.shape",frequence_array.shape)
concatenate_array=np.concatenate((depart,dep,title,askk),axis=1)
#print("concatenate_array",concatenate_array)
#print("concatenate_array",concatenate_array.shape)
data=DataFrame(concatenate_array,columns=["department","dep",'title','ask'])




    
#oldf.close()
#data.to_csv('E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/妇产科/计划生育/random2000.csv',index=False)

#train_data.close()
#validation_data.close()

#train_label.close()
#validation_label.close()

# 将训练集中拷贝到验证集的数据删除，训练数据
#df_data = pd.read_csv(open('E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random32000.csv',encoding='utf8'),skiprows=resultList)
df_data=dep_tit_ask_ans.drop(resultList)
df_data.to_csv('E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random22400.csv',index=False)
data.to_csv('E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random9600.csv',index=False)

# 将数据集中拷贝到验证集的数据删除，标签
#df_label = pd.read_csv('train_label.csv',skiprows=resultList)
#df_label.to_csv('train_label_delete.csv',index=False)


    


