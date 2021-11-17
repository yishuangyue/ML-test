#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/10 2:52 下午 
:@File : Bayes基础测试,用一些下列子梳理整个贝叶斯分布过程
:Version: v.1.0
:Description:
"""

from numpy import ones, array, log


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'i', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    """
    会创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet: 文档句子分词list
    :return: 全量词set
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 两个set的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """返回multi-hot向量"""
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word： %s is not in my Vocabulary!" % word)
    return returnVec


def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算侮辱类的概率
    pONum = ones(numWords)
    p1Num = ones(numWords)  # 初始化概率  如果其中一个概率值为0 , 那么最后的乘积也为0。为降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2
    pODenom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 向量相加
            p1Denom += sum(trainMatrix[i])
        else:
            pONum += trainMatrix[i]
            pODenom += sum(trainMatrix[i])
    plVect = log(p1Num / p1Denom)  # changetolog{)<3——n   在给定文档类别条件下词汇表中单词的出现概率
    p0Vect = log(pONum / pODenom)  # changetolog()每元素做除  由 于 太多很小的数相乘造成下溢问题。取对数解决
    return p0Vect, plVect, pAbusive


def classifyNB(vec2Classify, pOVec, p1Vec, pClass1):
    """
    计算不同类别下的条件概率
    :param vec2Classify: 输入文档或者句子的分词multi-hot list
    :param pOVec: 在给定文档类别条件下词汇表中单词的出现概率数组
    :param p1Vec:  在给定文档类别条件下词汇表中单词的出现概率数组
    :param pClass1: 类别为1时候的概率
    :return: 输入文档的类别
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 这里由于前面取对数 ln(a*b)=lna+lnb所以这里是+
    p0 = sum(vec2Classify * pOVec) + log(1.0 - pClass1)  # 办元素相乘
    print(p1, p0)
    if p1 > p0:
        return 1
    else:
        return 0

# main入口
if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()    # 返回分词结果以及类别list
    myVocabList = createVocabList(listOPosts)  # 返回全量词集合
    setOfWords2Vec(myVocabList, listOPosts[0]) # 返回multi-hot向量
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 下面给出属于侮辱性文档的概率以及给定文档类别条件下词汇表中单次的出现概率向量。
    pOV, p1V, pAb = trainNBO(trainMat, listClasses) # p0在给定文档类别条件下词汇表中单词的出现概率向量,
    testEntry = ['love', 'my', 'dalmation']  # 测试句子
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 返回测试句子的multi-hot向量
    print(testEntry, "classified as:", classifyNB(thisDoc, pOV, p1V, pAb))  # 计算不同类别下的条件概率
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classifiedas:', classifyNB(thisDoc, pOV, p1V, pAb))
