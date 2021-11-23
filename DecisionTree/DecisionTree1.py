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
八个参数：Criterion
两个随机性相关的参数（random_state，splitter）
五个剪枝参数（max_depth,min_samples_split，min_samples_leaf，max_feature，min_impurity_decrease）
一个属性：feature_importances_
四个接口：fit，score，apply，predict
无论如何，剪枝参数的默认值会让树无尽地生长，这些树在某些数据集上可能非常巨大，对内存的消耗也非常巨大。
所以如果你手中的数据集非常巨大，你已经预测到无论如何你都是要剪枝的，那提前设定这些参数来控制树的复杂性和大小会比较好。

"""
import joblib
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

# 建立模型
from Logistic.data_deal import data_deal


def fit_model(Xtrain, Xtest, Ytrain, Ytest, feature_name):
    clf = tree.DecisionTreeClassifier(criterion="entropy"
                                      , random_state=30
                                      , splitter="random"
                                      , max_depth=3
                                      #    ,min_samples_leaf=10
                                      #    ,min_samples_split=25
                                      )
    clf = clf.fit(Xtrain, Ytrain)

    # 3.1模型保存与加载
    joblib.dump(clf, model_path)  # 保存模型
    print("模型训练准确率：%s" % clf.score(Xtrain, Ytrain))
    tes_label = clf.predict(Xtest)  # 测试集的预测标签
    # lr_clf.predict_proba() # 返回每个样本的概率
    print("测试集精度：", accuracy_score(Ytest, tes_label))
    # 画一棵树
    dot_data = tree.export_graphviz(clf
                                    , feature_names=feature_name
                                    # , class_names=["琴酒", "雪莉", "贝尔摩德"]
                                    , class_names=["非正常", "正常"]
                                    , filled=True
                                    , rounded=True
                                    )
    print(graphviz.Source(dot_data))
    return clf


if __name__ == "__main__":
    model_path = '/Users/liting/Documents/python/Moudle/ML-test/Bayes/decisiontree1_model.m'
    # 探索数据
    # wine = load_wine()
    # # wine.data.shape
    # # wine.target
    # # 如果wine是一张表，应该长这样：
    # pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
    # # wine.feature_names
    # # wine.target_names
    # # 划分训练集和测试集
    # Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
    # feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒',
    #                 '脯氨酸']
    # clf = fit_model(Xtrain, Xtest, Ytrain, Ytest, feature_name)

    input_path = '/Users/liting/Documents/python/Moudle/ML-test/Logistic/ods_data.json'
    train_data, test_data, train_label, test_label = data_deal(input_path)
    train_label = train_label[:, 0]  # flatmap一下
    test_label = test_label[:, 0]
    feature_name = ['责令限期违规标志', '同法人状态非正常标志', '连续三月未申报标志', '身份证标志', '纳税人无销方信息标志'
        , '财务报表标志', '注册资本', '固定工人数', '法人年龄']
    clf = fit_model(train_data, test_data, train_label, test_label, feature_name)
    # 输出特征重要度
    clf.feature_importances_
    [*zip(feature_name, clf.feature_importances_)]
