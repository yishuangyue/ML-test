#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/11 3:23 下午 
:@File : DecisionTree
:Version: v.1.0
:Description:
"""
# 1构建数据集
import operator
from math import log


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing ', 'flippers']  # 是否能浮出水面，是否有脚踝，是否能生存
    return dataSet, labels


# 2、计算熵，为该类数据分类最好使用数据字典来保存分类结果
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 样本数量
    labelCounts = {}  # 创建记录不同分类标签结果多少的字典 {'yes': 2, 'no': 3}
    # 为所有可能分类保存
    # 该字典key：label value:label的数目
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 标签发生概率p(xi)的值
        shannonEnt -= prob * log(prob, 2)
    print("样本集{},的基础熵为：{}".format(dataSet,shannonEnt))
    return shannonEnt  # 熵


# 3、按特征划分数据返回，第axis个特征下的等于value分类的数据集
def splitDataSet(dataSet, axis, value):  # axis代表第几个特征 value为结果 splitDataSet(dataSet, 0, 0)
    # 避免修改原始数据集创建新数据集
    retDataSet = []
    # 抽取符合特征的数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 4、选择最优划分集 ——》第几个特征是最好的用于划分数据集的特征
# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 全部特征个数
    print("dataSet:{}".format(dataSet))
    baseEntropy = calcShannonEnt(dataSet)  # 基础熵
    print("第一步计算基础熵（基于不同分类下熵总和）：{}".format(baseEntropy))
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        print("featList:{}".format(featList))
        uniqueVals = set(featList)  # 建立列表同特征下不同回答
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            # print(dataSet, i, value)
            subDataSet = splitDataSet(dataSet, i, value)  # 划分
            prob = len(subDataSet) / float(len(dataSet))  # 同特征下不同回答所占总回答比率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 该特征划分下的信息熵
            print("第{}个特征下信息熵为{}".format(i,calcShannonEnt(subDataSet)))
            print("第{}个特征下值为{}后剩下特征的dataset为{}, 特征的经验条件熵{}, 该特征划分下的信息熵:{}".format(i,value,subDataSet, prob, newEntropy) )
        infoGain = float(baseEntropy - newEntropy)  # 信息增益
        print("信息增益:{}".format(infoGain))
        print("infoGain:{},{},{}".format(infoGain,bestInfoGain,baseEntropy))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    print("返回bestFeature：{}".format(bestFeature))
    return bestFeature


# 5、构造决策树（选择当下特征结果下，最多的那个分类）
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 6、构造递归决策树

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 保存标签
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止划分
        return classList[0]  # #当类别完全相同时则停止继续划分，直接返回该类的标签
    if len(dataSet[0]) == 1:  # #遍历完所有的特征时，仍然不能将数据集划分成仅包含唯一类别的分组 dataSet
        return majorityCnt(classList)  # 由于无法简单的返回唯一的类标签，这里就返回出现次数最多的类别作为返回值
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 获取最好的分类特征索引
    bestFeatLabel = labels[bestFeat]  # 获取该特征的名字
    myTree = {bestFeatLabel: {}}  # 这里直接使用字典变量来存储树信息，这对于绘制树形图很重要。
    del (labels[bestFeat])  # 使用完该特征划掉
    #
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 划分后特征组
        print(subLabels, value, bestFeatLabel)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        print("myTree：{}".format(myTree))
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


if __name__ == "__main__":
    # fr = open('play.tennies.txt')
    # lenses = [inst.strip().split(' ') for inst in fr.readlines()]
    # lensesLabels = ['outlook', 'temperature', 'huminidy', 'windy']
    dataSet, labels=createDataSet()
    lensesTree = createTree(dataSet, labels)
    # treePlotter.createPlot(lensesTree)
