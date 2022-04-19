# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:12:25 2021

@author: 23674
"""

import pandas as pd
import numpy as np
import jieba.posseg as pseg
from gensim.models import word2vec
import logging
import math
#停用词表
def StopWordsList(filepath):
    wlst = [w.strip() for w in open(filepath, 'r', encoding='utf8').readlines()]
    return wlst

#对句子分词
def seg_sentence(sentence, stop_words):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']#过滤数字m
#    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r']
    sentence_seged = pseg.cut(sentence)
    # sentence_seged = set(sentence_seged)
    outstr = []
    for word,flag in sentence_seged:
        # if word not in stop_words:
        if word not in stop_words and flag not in stop_flag:
            outstr.append(word)
    return outstr


#词向量化
def word2vector(sentences):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#    sentences = word2vec.Text8Corpus(u"E:/机器学习与医疗推荐/code/text classification/out_ask.txt")  # 加载语料
    n_dim=100
    model = word2vec.Word2Vec(sentences, size=n_dim,window=5, min_count=1,sg=1,hs=1)
#    corpus=model.save("E:/机器学习与医疗推荐/code/text classification/abs0.model")
    return model




#平均词向量
def get_sentence_matrix(model,new_keyword):
    sentences_matrix = []
    
    #平均特征矩阵
    
    word_matrix = np.zeros(model.vector_size)
    #得出各个词的特征向量，并形成一个矩阵，然后计算平均值，就得到该句子的特征向量
    for word in new_keyword:

        if word in model:
            word_matrix+=np.array(model[word])

    sentences_matrix=word_matrix/len(new_keyword)
    return sentences_matrix



#欧氏距离
def cosim(test, train):
    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(test)):
        sum+=test[i] * train[i]
        sq1+=pow(test[i],2)
        sq2+=pow(train[i],2)
    if math.sqrt(sq1) * math.sqrt(sq2) == 0:
        ED=0
    else:
        ED=float(sum)/(math.sqrt(sq1) * math.sqrt(sq2))
    

    return ED


def maxsim(simed,inde):

    x=simed.copy()
#    a=np.array(x)
    x.sort()
#    print(x)
    a=x[-1:]
    c=[]
    for b in a:
        ind=simed.index(b)
#        print(ind)
        i=inde[ind]
        c.append(i)
    
    print(c)
#    b=max(a)
#    print(a)
    return c    

    
    


if __name__ == '__main__':
#载入停用词典
    spPath = 'E:\机器学习与医疗推荐\pkuseg_st\stopwords.txt'

#载入文件
    trainpath = 'E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random22400.csv'
    testpath = 'E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/外科/test2400/random2400.csv'
    stop_words = StopWordsList(spPath)

    ##训练文件
    dep_tit_ask_ans = pd.read_csv(open(trainpath, encoding='utf8'), sep=',')        
    tit_ask=dep_tit_ask_ans['title'].copy()+dep_tit_ask_ans['ask'].copy()
    dep=dep_tit_ask_ans["dep"].copy()
    depart=dep_tit_ask_ans["department"].copy()
    texts = [seg_sentence(seg, stop_words) for seg in tit_ask]

    ##测试文件
    test_file=pd.read_csv(open(testpath, encoding='utf8'), sep=',')
    test_tit_ask=test_file['title'].copy()+test_file['ask'].copy()
    keywords=[seg_sentence(seg, stop_words) for seg in test_tit_ask]
    test_set=[i for w in keywords for i in w ]

    model=word2vector(texts)   #
 #   model=word2vec.Word2Vec.load("E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/结果集/random22400_sample_100_53_cbow.model")
    
    wordnp=[]
    for text in texts:
        matrix_text=get_sentence_matrix(model,text)
        wordnp.append(matrix_text)
    
    result1=[]
    result2=[]
    for t in range(len(keywords)):
        
        result1.append([])
        result2.append([])
        matrix=get_sentence_matrix(model,keywords[t])   #model 
#        print("test%d"%(keywords.index(keyword)),matrix)
##        print('\n')
        simi=[]
        simed=[]
        inde=[]
        for wordn in wordnp:
            sim=cosim(matrix, wordn)
#            print('matrix%d 与 matrix_text%d 相似度为：%f' % (keywords.index(keyword)+1,texts.index(text)+1,ed))
            simi.append(sim)
        for ind,val in enumerate(simi):            
            if val >0.8:
                inde.append(ind)
                simed.append(val)
#        print(t,simed)

        if simed==[]:
            pass
        else:
            maxsimi=maxsim(simed,inde)
#            print(maxsimi)

            for s in maxsimi:            
               result1[t]=dep[s]
               result2[t]=depart[s]
               print('matrix%d 与 matrix_text%d 相似度为：%f' % (t+1,s+1,simi[s]))
#               if simi[s] == 1.0:
#                   print(test_tit_ask[t],tit_ask[s])

    fin1=[list(set(i)) for i in result1]
    fin2=[list(set(i)) for i in result2]            
    test_file.insert(1,"finaly_result1",result1)
    test_file.insert(2,"finaly_result2",result2)
#    print(test_file.finaly_result1,test_file.finaly_result2)
    test_file.to_csv("E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/词向量结果/外科random2400_0.8_100_53_cbow.csv",index=False)

