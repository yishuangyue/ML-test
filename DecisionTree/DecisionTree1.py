#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/18 2:08 下午 
:@File : DecisionTree1
:Version: v.1.0
:Description:
基于ID3算法的信息增益来实现的
"""
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

# 探索数据
wine = load_wine()
# wine.data.shape
# wine.target
# 如果wine是一张表，应该长这样：
pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
wine.feature_names
wine.target_names
#划分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
#建立模型
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain, Ytrain)
clf.predict(Xtest)
score = clf.score(Xtest, Ytest) #返回预测的准确度
# 画一棵树
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']

dot_date = tree.export_graphviz(clf
                               ,feature_names = feature_name
                                ,class_names=["琴酒","雪莉","贝尔摩德"]
                               ,filled=True
                               ,rounded=True)
graph = graphviz.Source(dot_date)
graph

# 输出特征重要度
clf.feature_importances_
[*zip(feature_name,clf.feature_importances_)]

