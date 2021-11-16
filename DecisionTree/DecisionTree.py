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
    numEntries = len(dataSet)
    labelCounts = {}  # 创建记录不同分类标签结果多少的字典
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
    return shannonEnt  # 熵


# 3、按特征划分数据
def splitDataSet(dataSet, axis, value):  # axis代表第几个特征 value为结果
    # 避免修改原始数据集创建新数据集
    retDataSet = []
    # 抽取符合特征的数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 4、选择最优划分集
# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 全部特征
    baseEntropy = calcShannonEnt(dataSet)  # 基础熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 建立列表同特征下不同回答
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 划分
            prob = len(subDataSet) / float(len(dataSet))  # 同特征下不同回答所占总回答比率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 该特征划分下的信息熵
        infoGain = float(baseEntropy - newEntropy)  # 信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 5、构造决策树

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.key(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 6、构造递归决策树

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 保存标签
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止划分
        return classList[0]  # 返回出现次数最多的标签
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 使用完该特征划掉
    # 得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 划分后特征组
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
