#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/15 6:33 下午 
:@File : Bayes1 在贝叶斯0的基础上进阶版，实现词袋模型,相比于bayes0中函数，它与函数從匕(^时0 =乜 祝 扣 （）几乎
完全相同，唯一不同的是每当遇到一个单词时，它会增加词向量中的对应值，而不只是将对应的 数值设为1。
步骤：
(1)收集数据：提供文本文件。
(2)准备数据：将文本文件解析成词条向量。
(3)分析数据：检查词条确保解析的正确性。
(4)训练算法：使用我们之前建立的trainNBO函数。
(5)测试算法：使用classifyNB，并且构建一个新的测试函数来计算文档集的错误率。
(6)使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上。
:Version: v.1.0
:Description:
"""

import re, random
from numpy import ones, array, log


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
            returnVec[vocabList.index(word)] += 1
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


def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList, classList, fullText = [], [], []
    for i in range(1, 26):
        wordList = textParse(open("email/spam/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open("email/ham/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)  # 导入并解析文本文件
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = [i for i in range(50)]
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
        pOV, plV, pSpam = trainNBO(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), pOV, plV, pSpam) != classList[docIndex]:  # ：对测试集分类©
            errorCount += 1
    print('the errorrateis:1,float(errorCount)/len(testSet)')


# main入口
if __name__ == '__main__':
    # mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
    # regEx = re.compile('\\W+')  # 过滤除单词数字以外的字符
    # listOfTokens = regEx.split(mySent)
    # listOfTokens1 = [tok.lower() for tok in listOfTokens if len(tok) > 0]  # 将单词转换为小写
    spamTest()
