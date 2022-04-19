# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:09:13 2021

@author: 23674
"""

import pandas as pd
import numpy as np
import jieba.posseg as pseg
from gensim.models import word2vec

from gensim import corpora, models, similarities

#停用词表
def StopWordsList(filepath):
    wlst = [w.strip() for w in open(filepath, 'r', encoding='utf8').readlines()]
    return wlst

#对句子分词、去停用词
def seg_sentence(sentence, stop_words):
    # stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']#过滤数字m
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r']
    sentence_seged = pseg.cut(sentence)
    # sentence_seged = set(sentence_seged)
    outstr = []
    for word,flag in sentence_seged:
        # if word not in stop_words:
        if word not in stop_words and flag not in stop_flag:
            outstr.append(word)
    return outstr



def bow_sim(index,sim):
    result_sim=[]
    result_index = []
    result_dep=[]
    result_depart=[]
    tfidf_bow_sim_numpy=sim
    for j in range(len(tfidf_bow_sim_numpy)):      #36
        if tfidf_bow_sim_numpy[j]> 0.5   :
            result_index.append(j)
            result_sim.append(tfidf_bow_sim_numpy[j])
#    print(j,result_index,result_sim)
#            result_list.append(dep_txt[j])
    if result_sim == [] :
        pass
    else:
        x=result_sim.copy()        
        x.sort()
        res=x[-1:]
        for b in res:
            
            simin=result_sim.index(b)
            depin=result_index[simin]
#            print(result_sim,b)
            result_dep=dep_txt[depin]
            result_depart=depart_txt[depin]
            print('test%d 与 train%d 相似度为：%.2f' % (index+1,depin+1,result_sim[simin]))
        

    return result_dep,result_depart





if __name__ == '__main__':
    #载入停用词典、训练文件、测试文件
    spPath = 'E:\机器学习与医疗推荐\pkuseg_st\stopwords.txt'
    trainpath = 'E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random22400.csv'
    testpath = 'E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/妇产科/test2400/random2400.csv'


    stop_words = StopWordsList(spPath)

    ##训练文件
    dep_tit_ask_ans = pd.read_csv(open(trainpath, encoding='utf8'), sep=',')    
    tit_ask=dep_tit_ask_ans['title'].copy()+dep_tit_ask_ans['ask'].copy()
    dep=dep_tit_ask_ans["dep"].copy()
    depart=dep_tit_ask_ans["department"].copy()
    tit_ask_data=pd.DataFrame(tit_ask)
    texts = [seg_sentence(seg, stop_words) for seg in tit_ask]
    orig_txt = [seg for seg in tit_ask]
    dep_txt=[seg for seg in dep]
    depart_txt=[seg for seg in depart]
    ##测试文件
    test_file=pd.read_csv(open(testpath, encoding='utf8'), sep=',')
    test_tit_ask=test_file['title'].copy()+test_file['ask'].copy()
    keywords=[seg_sentence(seg, stop_words) for seg in test_tit_ask]
    #生成字典
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id.keys())
    #生成语料
    corpus = [dictionary.doc2bow(text) for text in texts]
    #定义TFIDF模型
    tfidf = models.TfidfModel(corpus)
    #用语料训练模型并生成TFIDF矩阵
    corpus_tfidf = tfidf[corpus]
    result1=[]
    result2=[]
    the_empty=[]
    for i in range(len(keywords)):
        result1.append([])
        result2.append([])
        kw_vector = dictionary.doc2bow(keywords[i])
        #生成余弦相似度索引
        index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=feature_cnt)        
        sim = index[tfidf[kw_vector]]
        d1,d2=bow_sim(i,sim)
        result1[i]=d1
        result2[i]=d2
        
        
    fin1=[list(set(i)) for i in result1]
    fin2=[list(set(i)) for i in result2]
    test_file.insert(1,"finaly_result1",result1)
    test_file.insert(2,"finaly_result2",result2)
#    print(test_file.finaly_result1,test_file.finaly_result2)
    test_file.to_csv("E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/词袋模型/妇产科random2400_0.8.csv",index=False)
       