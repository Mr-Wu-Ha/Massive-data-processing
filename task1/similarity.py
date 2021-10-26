import numpy as np
import time
from context import tfidf
from context import cosine_similarity
from context import preprocess
from context import essaysplit
from context import con_vector


#预处理
p_time_start=time.time()
inputpath='199801_clear.txt'
stopwordlist='StopWordList.txt'
inputword,stop_list=preprocess(inputpath,stopwordlist)
essays,count_l=essaysplit(inputword,stop_list)
print("预处理结束")
p_time_end=time.time()

#计算tfidf值以及耗时,取每篇文章前15个单词的tfidf
t_time_start=time.time()
tfidf_a=[]
for i,count in enumerate(count_l):
    s={}
    for word in count:
        tfi=tfidf(word,count,count_l)
        s.update({word:tfi})
    sorted_s = sorted(s.items(), key=lambda x: x[1], reverse=True)
    tfidf_a.append(dict(sorted_s[0:15]))
print("tfidf值计算结束")
t_time_end=time.time()

#计算两两文章的余弦相似度，得到相似矩阵
d_time_start=time.time()
sim_mat=np.zeros((len(tfidf_a),len(tfidf_a)))
for i in range(0,len(tfidf_a)):
    for j in range(i,len(tfidf_a)):
        list_i,list_j=con_vector(tfidf_a[i],tfidf_a[j])
        sim_mat[i][j]=cosine_similarity(list_i,list_j)
        sim_mat[j][i]=sim_mat[i][j]
print("余弦值计算结束")
d_time_end=time.time()

print("预处理花费:"+str(p_time_end-p_time_start)+"s")
print("计算所有文章的tfidf值花费:"+str(t_time_end-t_time_start)+"s")
print("计算所有文章的余弦相似度花费:"+str(d_time_end-d_time_start)+"s")

#保存相似矩阵
with open('similarity_mat.txt','w') as outp:
    for row in range(len(tfidf_a)):
        outp.write(str(sim_mat[row])+'\n')







