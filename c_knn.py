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
from gensim import corpora, models, similarities
import math
from sklearn import neighbors


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
#        words = new_keword[index].split(" ")
        #得出各个词的特征向量，并形成一个矩阵，然后计算平均值，就得到该句子的特征向量
    for word in new_keyword:
            #当前词是在word2vec模型中，self.model为词向量模型
        if word in model:
            word_matrix+=np.array(model[word])
#                word_matrix.append(np.array(model[word]))
            #将words_matrix求均值
#        featrue = averageVector(many_vectors=word_matrix,
#                                column_num=model.vector_size)
#        sentences_matrix.append(featrue)
    sentences_matrix=word_matrix/len(new_keyword)
    return sentences_matrix





#欧氏距离
def euclideandistances(test, train):
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
    
#    ED = np.sqrt(np.sum(np.square(train - test)))
    return ED



'''
#欧氏距离
def euclideandistances(test, train):
    ED = np.sqrt(np.sum(np.square(train - test)))
    return 1-ED
''' 




'''
def maxsim(simi):
    x=simi.copy()
#    a=np.array(x)
    x.sort()
#    b=max(a)
#    print(a)
    return x[-2:]
'''

def maxsim(simed,inde):
#    y2 = model.most_similar(u"糖尿病", topn=2)  # 20个最相关的
#    print (u"和糖尿病最相关的词有：\n")
#    for item in y2:
#        print (item[0], item[1])
#        print ("--------\n")
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
    spPath = 'E:\机器学习与医疗推荐\pkuseg_st\stopwords.txt'
#    trainpath = 'E:\代码及数据集\Chinese-medical-dialogue-data-master\Chinese-medical-dialogue-data-master\样例小样本_内科5000-6000.csv'
#    testpath = 'E:\代码及数据集\Chinese-medical-dialogue-data-master\Chinese-medical-dialogue-data-master\内科小样本_test.csv'


    trainpath = 'E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/random22400.csv'
    testpath = 'E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/妇产科/test2400/random2400.csv'
    stop_words = StopWordsList(spPath)
#    keyword = '胸闷喘息有点困难胃部胀气不舒服，'
    # 1、将【文本集】生产【分词列表】
    ##训练文件
    dep_tit_ask_ans = pd.read_csv(open(trainpath, encoding='utf8'), sep=',')    
    
    tit_ask=dep_tit_ask_ans['title'].copy()+dep_tit_ask_ans['ask'].copy()
#    tit_ask=dep_tit_ask_ans['title'].copy()
    dep=dep_tit_ask_ans["dep"].copy()   #二级科室
    dep_labels={'耳鼻喉科':10,'营养保健科':11, '神经内科':12, '内科':13,
                '产科':20,'妇产科':21,'计划生育':22,'生殖医学科':23,
                '消化科':30,'神经科':31,'呼吸科':32,'内分泌科':33,
                '肛肠':40,'神经脑外科':41,'泌尿科':42,'普通外科':43}
    d1_k=dep_labels.keys()
    d1_v=dep_labels.values()
    dep_lab=[]
    for x in dep:
        d_l=dep_labels[x]
        dep_lab.append(d_l)
    depart=dep_tit_ask_ans["department"].copy()   #一级科室
    depart_labels={'儿科':1,'妇产科':2,'内科':3,'外科':4}
    d2_k=depart_labels.keys()
    d2_v=depart_labels.values()
#    print(type(d2_k),d2_v)
    depart_lab=[]
    for x in depart:
        d_l=depart_labels[x]
        depart_lab.append(d_l)
#    tit_ask_data=pd.DataFrame(tit_ask)
    texts = [seg_sentence(seg, stop_words) for seg in tit_ask]
#    orig_txt = [seg for seg in tit_ask]
#    dep_txt=[seg for seg in dep]
#    train_set=[i for w in texts for i in w ]##
    ##测试文件
    test_file=pd.read_csv(open(testpath, encoding='utf8'), sep=',')
    test_tit_ask=test_file['title'].copy()+test_file['ask'].copy()
#    test_tit_ask=test_file['title'].copy()
    keywords=[seg_sentence(seg, stop_words) for seg in test_tit_ask]
    test_set=[i for w in keywords for i in w ]
    #逐句分析
#    print('k',keywords)
#    print(train_set)
    
#    model=word2vector(texts)   #
    model=word2vec.Word2Vec.load("E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/结果集/random22400_sample_100_53_cbow.model")
    knn1 = neighbors.KNeighborsClassifier(n_neighbors=6)
    knn2 = neighbors.KNeighborsClassifier(n_neighbors=6)
    
    wordnp=[]
    for text in texts:
        matrix_text=get_sentence_matrix(model,text)
        wordnp.append(matrix_text)
        
    knn1.fit(np.array(wordnp),np.array(dep_lab))
    knn2.fit(np.array(wordnp),np.array(depart_lab))
    
    testnp=[]
    for t in keywords:
        matrix_test=get_sentence_matrix(model,t)
        testnp.append(matrix_test)
    predict_lab1=knn1.predict(testnp)        
    predict_lab2=knn2.predict(testnp)
    
#    print(predict_lab2)
    predict_l1=[]
    for y1 in predict_lab1:
        get_depart_index = list(d1_v).index(y1)
        v=list(d1_k)[get_depart_index]
        predict_l1.append(v)
    predict_l2=[]
    for y2 in predict_lab2:
        get_depart_index = list(d2_v).index(y2)
        v=list(d2_k)[get_depart_index]
        predict_l2.append(v)
#    print(predict_l)
    test_file.insert(1,"finaly_result1",predict_l1)
    test_file.insert(2,"finaly_result2",predict_l2)
    test_file.to_csv("E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/KNN结果/妇产科random2400_100_53_knn_6.csv",index=False)


