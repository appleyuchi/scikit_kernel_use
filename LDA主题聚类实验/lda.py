# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import jieba


def fenci(path,path_fenci):
    with open(test0) as f:
        document = f.read()
        document_decode = document.decode('utf-8')
        document_cut = jieba.cut(document_decode)
        #print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
        result = ' '.join(document_cut)
        result = result.encode('utf-8')
        with open(path_fenci, 'w') as f2:
            f2.write(result)
    f.close()
    f2.close()

def data_read(path_fenci):
    with open(path_fenci) as f3:
        return f3.read()





if __name__ == '__main__':
    test0='./nlp_test0.txt'
    test0_fenci='./nlp_test1.txt'
    fenci(test0,test0_fenci)
    res1=data_read(test0_fenci)
###################################
    test0='./nlp_test2.txt'
    test0_fenci='./nlp_test3.txt'
    fenci(test0,test0_fenci)
    res2=data_read(test0_fenci)
###################################
    test0='./nlp_test4.txt'
    test0_fenci='./nlp_test5.txt'
    fenci(test0,test0_fenci)
    res3=data_read(test0_fenci)

    jieba.suggest_freq('沙瑞金', True)
    # 使用suggest_freq(segment, tune=True) 可调节单个词语的词频,使其(或不能)被分出来
    jieba.suggest_freq('易学习', True)
    jieba.suggest_freq('王大路', True)
    jieba.suggest_freq('京州', True)
    #从文件导入停用词表
    stpwrdpath = "stop_words.txt"
    stpwrd_dic = open(stpwrdpath, 'rb')
    stpwrd_content = stpwrd_dic.read()
    #将停用词表转换为list  
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()

    corpus = [res1,res2,res3]
    cntVector = CountVectorizer(stop_words=stpwrdlst)
    cntTf = cntVector.fit_transform(corpus)
    print cntTf
    lda = LatentDirichletAllocation(n_topics=2, max_iter=5,#需要自己指定类别数
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    docres = lda.fit_transform(cntTf)
    print lda.components_
    print docres
