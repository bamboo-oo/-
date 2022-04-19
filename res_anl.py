# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:34:57 2021

@author: 23674
"""

import pandas as pd
#载入待分析文件
res_an="E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/方法/top/妇产科random2400_0.8_nodef_top5.csv"
open_res=pd.read_csv(open(res_an, encoding='utf8'), sep=',')

dep1 = open_res['dep'].tolist()
dep2 = open_res['department'].tolist()
dd=list()


fin_res1 = open_res['finaly_result1'].tolist()
fin_res2 = open_res['finaly_result2'].tolist()
p=len(fin_res1)

count=0
c=0
num=range(0,2400)
for i in num:
#    print(i,fin_res[i],fin_res[i])

    if dep1[i] == fin_res1[i]:
#        print(i,dep[i],fin_res[i])
        count=count+1

        
a=0
b=0
for i in num:
    if dep2[i] == fin_res2[i]:
        c=c+1
#        print(dep[i],fin_res[i])

        
    
for s in num:
    
#    print(fin_res[s])
    if fin_res1[s] == '[]':
        
        b=b+1
        
    else:    
        list1= fin_res1[s].split(",")
        for t in list1:
#            print(t)
            a=a+1
    
    
#print(dep[0])
#print("总条数，正确条数，结果数分别是：",p,count,a)
print(c,count,a,b,len(num))
