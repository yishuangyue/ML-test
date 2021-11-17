#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/16 3:41 下午 
:@File : Bayes2
测试高斯朴素贝叶斯 分类器假设每个标签的数据都服从简单的高斯分布。
直接调用sklearn包运行
其中GaussianNB就是先验为高斯分布的朴素贝叶斯，
MultinomialNB就是先验为多项式分布的朴素贝叶斯
而BernoulliNB就是先验为伯努利分布的朴素贝叶斯
:Version: v.1.0
:Description:
"""
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from svm import svm


def mian_test():
    iris = datasets.load_iris()  # 鸢尾花测试数据集
    clf = MultinomialNB()
    clf.fit(iris.data, iris.target)  # 训练 （数据，类别）
    y_pred = clf.predict(iris.data)  # 预测
    print("多项分布朴素贝叶斯，样本总数： %d 错误样本数 : %d"
          % (iris.data.shape[0], (iris.target != y_pred).sum()))

def mian_two():
    input_path = '/Users/liting/Documents/python/Moudle/ML-test/svm/formatData.txt'
    train_data, test_data, train_label, test_label = svm.data_deal(input_path)
    train_data = train_data[:, 1:]
    train_label = train_label[:, 0]  # flatmap一下
    clf = BernoulliNB()
    clf.fit(train_data, train_label)  # 训练 （数据，类别）
    y_pred = clf.predict(train_data)  # 预测
    print(y_pred)
    print("多项分布朴素贝叶斯，样本总数： %d 错误样本数 : %d"
          % (train_data.shape[0], (train_label != y_pred).sum()))


# main入口

if __name__ == '__main__':
    # mian_test()
    mian_two()
