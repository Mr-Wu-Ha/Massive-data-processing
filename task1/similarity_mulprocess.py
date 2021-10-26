import numpy as np
import time
from context import tfidf
from context import cosine_similarity
from context import preprocess
from context import essaysplit
from context import con_vector
import multiprocessing as mp


def mul_tfidf(i,count, count_l,shared_d):
    s = {}
    for word in count:
        tfi = tfidf(word, count, count_l)
        s.update({word: tfi})
    sorted_s = sorted(s.items(), key=lambda x: x[1], reverse=True)
    shared_d[i]=dict(sorted_s[0:15])


def mul_rcos(i, tfidf_a, p_mat):
    for j in range(i, len(tfidf_a)):
        list_i, list_j = con_vector(tfidf_a[i], tfidf_a[j])
        p_mat[i][j] = cosine_similarity(list_i, list_j)


def main(argv=None):
    # 预处理
    p_time_start = time.time()
    inputpath = '199801_clear.txt'
    stopwordlist = 'StopWordList.txt'

    inputword, stop_list = preprocess(inputpath, stopwordlist)

    essays, count_l = essaysplit(inputword, stop_list)
    print("预处理结束")
    p_time_end = time.time()
    print("预处理花费:" + str(p_time_end - p_time_start) + "s")

    # 计算tfidf值以及耗时,取每篇文章前15个单词的tfidf
    t_time_start = time.time()
    tfidf_a = []
    shared_d=mp.Manager().dict()
    po=mp.Pool(32)
    for i, count in enumerate(count_l):
        po.apply_async(mul_tfidf,args=(i,count,count_l,shared_d))
    po.close()
    po.join()
    print("tfidf值计算结束")
    t_time_end = time.time()
    print("计算所有文章的tfidf值花费" + str(t_time_end - t_time_start) + "s")
    for i in range(len(shared_d)):
        tfidf_a.append(dict(shared_d[i]))
    #print(tfidf_a)

    # 计算两两文章的余弦相似度，得到相似矩阵
    d_time_start = time.time()
    sim_mat = []
    p_mat = mp.Manager().list()
    po2 = mp.Pool(32)
    for i in range(0, len(tfidf_a)):
        po2.apply_async(mul_rcos, args=(i, tfidf_a, p_mat))
    po2.close()
    po2.join()
    print("余弦值计算结束")
    d_time_end = time.time()
    for i in range(len(p_mat)):
        sim_mat.append(p_mat[i])
    print("计算所有文章的余弦相似度花费:" + str(d_time_end - d_time_start) + "s")


    # 保存相似矩阵
    with open('similarity_mat_mulp.txt', 'w') as outp:
        for row in range(len(p_mat)):
            outp.write(str(sim_mat[row]) + '\n')

if __name__=='__main__':
    main()
