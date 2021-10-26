import pandas as pd
import re
import math
import numpy as np
from collections import Counter


#计算标准的TF-IDF
def tfidf(word,count,count_l):
    tf = count[word]/sum(count.values())
    c_sum = 0
    for c in count_l:
        if word in c:
            c_sum += 1
    idf = 1 + math.log(len(count_l) / c_sum)
    return tf*idf


#构建文本向量
def con_vector(tf_idfDicA,tf_idfDicB):
    listA = []
    listB = []
    for wordA in tf_idfDicA:
        listA.append(tf_idfDicA[wordA])
        if wordA not in tf_idfDicB:
            listB.append(0)
        else:
            listB.append(tf_idfDicB[wordA])
    for wordB in tf_idfDicB:
        if wordB not in tf_idfDicA:
            listB.append(tf_idfDicB[wordB])
            listA.append(0)
    return listA, listB


#计算余弦相似度
def cosine_similarity(vec1 , vec2):
    res_up=np.dot(vec1,vec2)
    res_d= (np.dot(vec1 , vec1) ** 0.5) * (np.dot(vec2 , vec2)** 0.5)
    if not res_up or not res_d:
        return 0
    else:
        return round(res_up/res_d,4)


#词表与停词文件的读入，预处理
def preprocess(inputfile,stopfile):
    input=open(inputfile,'r',encoding='gbk')
    inputword=input.read()
    input.close()

    stopword=pd.read_csv(stopfile,sep='\n',encoding='utf-8',names=['stopword'],index_col=False,quoting=3)
    stop_list=stopword['stopword'].tolist()
    return inputword,stop_list


#将词表中不同文章区分出来
def essaysplit(inputword,stop_list):
    essays = []
    file_i = inputword.split('\n\n')
    for file in file_i:
        essays.append(file)
    essay_w=[]
    count_l=[]
    for e in essays:
        essay_wl=[]
        essay=e.split('\n')
        for sentences in essay:
            sentence=sentences.split(' ')
            del sentence[0]
            for word in sentence:
                word = re.sub(r'/[A-Za-z]*','',word)
                if word not in stop_list and word != '':
                    essay_wl.append(word)
        count=Counter(essay_wl)
        count_l.append(count)
        essay_str=','.join(essay_wl)
        essay_str=re.sub(',','',essay_str)
        essay_w.append(essay_str)
    return essay_w,count_l

