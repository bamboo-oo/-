# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:47:29 2021

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

#对句子分词,去停用词
def seg_sentence(sentence, stop_words):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']#过滤数字m
#    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r']
    sentence_seged = pseg.cut(sentence)
    outstr = []
    for word,flag in sentence_seged:
        if word not in stop_words and flag not in stop_flag:
            outstr.append(word)
    return outstr


#词向量化
def word2vector(sentences):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#    sentences = word2vec.Text8Corpus(u"E:/机器学习与医疗推荐/code/text classification/out_ask.txt")  # 加载语料
    n_dim=100
    model = word2vec.Word2Vec(sentences, size=n_dim,window=5, min_count=3,sg=1,hs=1)
    
#保存向量化结果
#    corpus=model.save("E:/机器学习与医疗推荐/code/text classification/abs0.model")
    return model

def train_idf(doc_list):
    idf_dic={}
    tt_count=len(doc_list)
    
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word]=idf_dic.get(word,0.0)+1.0
    
    for k,v in idf_dic.items():
        idf_dic[k]=math.log(tt_count/(1.0+v))
        
    print("tt_count"+str(tt_count))
    default_idf=math.log(tt_count/(1.0))
    return idf_dic,default_idf



def cut_sentence(keyword):
    fm=0
    toplist=[]
    for word in keyword:
        if word in model:
#            print(word)
            fm+=1


    #扩展词向量
    if fm<8:
        for word in keyword:
            if word in model:
                new_word=model.most_similar(u"%s"%(word), topn=1)
                for item in new_word:
                    toplist.append(item[0])
                    keyword=keyword+toplist
                    
    return keyword


#平均词向量
def get_sentence_matrix(model,new_keyword):
    sentences_matrix = []
    #平均特征矩阵
#    new_keyword=cut_sentence(keyword)
    word_matrix=np.zeros(model.vector_size)       
    for word in new_keyword:
        if word in model:
            word_matrix+=np.array(model[word])

    sentences_matrix=word_matrix/len(new_keyword)
    return sentences_matrix


def def_ma(model,texts):
    num=0
    def_ma=np.zeros(model.vector_size)
    for text in texts:
        for word in text:
            if word in model:
                num+=1
                def_ma+=np.array(model[word])
    return def_ma/num


#余弦距离
def cosim(test, train):
    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(test)):
        sum+=test[i] * train[i]
        sq1+=pow(test[i],2)
        sq2+=pow(train[i],2)
    if math.sqrt(sq1) * math.sqrt(sq2)==0:
        simi=0
    else:
        simi=float(sum)/(math.sqrt(sq1) * math.sqrt(sq2))
#    ED = np.sqrt(np.sum(np.square(train - test)))
    return simi

def MD(x, y):
#汉明距离
    d=sum(x_ch != y_ch for x_ch, y_ch in zip(x, y))
    print(d)
    return d

#切比雪夫距离    
#    x = np.array(x)
#    y = np.array(y)
#    return np.max(np.abs(x-y))


#闵氏距离
#    zipped_coordinate = zip(x, y)
#    d=math.pow(np.sum([math.pow(np.abs(i[0]-i[1]), p) for i in zipped_coordinate]), 1/p)
#    print(d)
#    return d





def ED(test, train):
#    X=np.vstack([test,train])

#方法一：根据公式求解
#    sk=np.var(X,axis=0,ddof=1)
#    ed=np.sqrt(((test - train) ** 2 /sk).sum())
    ed=np.sqrt(np.sum(np.square(test-train)))
    ed1=1/ed
    print('ed',ed,'\n','ed1',ed1)
    return ed1*0.6


def maxsim(matrix,simed,inde):
    x=simed.copy()
#    a=np.array(x)
    x.sort()
#    print(x)
    a=x[-6:]  #相似度topn(min-->max)
    a=set(a)
    c=[]
    d1=[]
    d2=[]
#    print(a)   #相似值
#    c_d=[]
    for b in a:
        ind=simed.index(b)  #相似度topa的索引值        
        i=inde[ind]        
        c.append(i)
#        c_d.append(cos_ed) 
        d1.append(depart[i])
        d2.append(dep[i])
#    dict_c_d=zip(c,c_d)
#    print(d1)
#    print(c)
#    result_pd=pd.DataFrame([])    
#    result_pd.insert(0,'index',c)
#    result_pd.insert(1,'depart',d1)
#    result_pd.insert(2,'dep',d2)
#    result_pd.insert(3,'sum_sim',a)
#  
#    print(result_pd)
#    del_c=result_pd.groupby(['depart']).size()
    
    
    
    
    fin_d1=set(d1)    #去除重复值，得到结果集
#    len_fin_d1=len(fin_d1)     #结果集的长度
    dep_list=dict()
#    dict.setdefault(key,[]).append(value)
    for i in fin_d1:
#        dep_list[i] = []
        for j in range(len(d1)):
#            print('i,d1[j]',i,d1[j])
            if d1[j] == i:
                dep_list.setdefault(i,[]).append(c[j])
#    print('dep_list',dep_list)
    
    res_list=[]
    sum_dic=0
    for l in dep_list.values():
        res_list.append(len(l))
        sum_dic+=len(l)
#        print('sum_dic',sum_dic)
    res_list.sort()
#    print('res_list',res_list)
    lar_res=res_list[len(res_list)-1]
#    print('lar_res',lar_res)
    fin_dep_list = []
#    sum_weight=[]
    for index,value in dep_list.items():
#        print('len(value)',len(value))
#        l=len(value)
#        d_weight=(l/sum_dic)*0.2
#        for x in c_d:
#            sum_w=d_weight+x
#        fin_dep_list.setdefault(index,[]).append(sum_w)
#    print('fin_dep_list',fin_dep_list)
        if len(value) == lar_res:
            fin_dep_list+=dep_list[index]
#            print('fin_dep_list',fin_dep_list)
    global s_w
    s_w=dict()
    for f in fin_dep_list:
#        ed_weight=MD(matrix,wordnp[f])
        cos_weight=simi[f]
#        sum_weight=ed_weight+cos_weight
        sum_weight=cos_weight
        s_w.setdefault(f,[]).append(sum_weight)
    
#    print(s_w)

    r_d=pd.DataFrame(s_w)
    top_s=r_d.idxmax(1).ix[0]
#    print(top_s)
#    print(s_w.get(top_s))
#    print(r_d)
#    top1_dep=fin_dep_list[-1:]
#    print(top1_dep)
#    b=max(a)
#    print(a)
    return top_s,s_w.get(top_s)    

#def density(dep_list,a):
#    for index,value in dep_list.items():
#        d_weight=len(value)/len(a)
#        
    


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

    
#    model=word2vector(texts)   
    model=word2vec.Word2Vec.load("E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/结果集/random22400_sample_100_53.model")

    words = list(model.wv.vocab.keys())
    vector = model[words]
    
    wordnp=[]
    for text in texts:
        matrix_text=get_sentence_matrix(model,text)
        wordnp.append(matrix_text)
   
    result1=[]
    result2=[]
    for t in range(len(keywords)):
#    for t in range(18,23):    
        result1.append([])
        result2.append([])

        matrix=get_sentence_matrix(model,keywords[t])   

        simi=[]
        simed=[]
        inde=[]
        for wordn in wordnp:

            sim=cosim(matrix, wordn)
#            print('matrix%d 与 matrix_text%d 相似度为：%f' % (keywords.index(keyword)+1,texts.index(text)+1,ed))
            simi.append(sim)
#            for i in simi:
        for ind,val in enumerate(simi):            
            if val >0.8:
                inde.append(ind)
                simed.append(val)
#        print(t,simed)

        if simed==[]:
            pass
        else:
            maxsimi,s=maxsim(matrix,simed,inde)
#            print(maxsimi)

#            for s in maxsimi:            
            result1[t]=dep[maxsimi]
            result2[t]=depart[maxsimi]  
            for i in s:
                print('matrix%d 与 matrix_text%d 相似度为：%f' % (t+1,maxsimi+1,i))
#    fin1=[list(set(i)) for i in result1]
#    fin2=[list(set(i)) for i in result2]            
    test_file.insert(1,"finaly_result1",result1)
    test_file.insert(2,"finaly_result2",result2)
#输出推荐结果
    test_file.to_csv("E:/代码及数据集/Chinese-medical-dialogue-data-master/Chinese-medical-dialogue-data-master/random/新方法/外科random2400_re_6.csv",index=False)
            





