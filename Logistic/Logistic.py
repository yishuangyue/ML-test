#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/18 9:53 上午 
:@File : Logistic
:Version: v.1.0
:Description:
基于：Logistic回归和Sigmoid函数的分类
步骤：
(1)收集数据：采用任意方法收集数据。
(2)准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
(3)分析数据：采用任意方法对数据进行分析。
(4)训练算法：大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
(5)测试算法：一旦训练步驟完成，分类将会很快。
(6)使用算法：首先，我们需要输入一些数据，并将其转换成对应的结构化数值；
接着，基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定它们属于 哪个类别.，在这之后，我们就可以夺输出的类别上做一些其他分析工作。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 能正确显示正负号

def fit_model0():
    # 数据处理
    # 加载数据
    data = np.loadtxt("/Users/liting/Documents/python/Moudle/ML-test/svm/formatData.txt", dtype=str, delimiter=',')
    # 切分
    # 参数一，被切分的矩阵
    # 参数二代表如何切分，[-1]代表-1之前的归为第一个返回值，其后归为第二个返回值
    # 参数三，axis=0是横向切分，切分样本；axis=1是纵向切分，切分的是特征
    x, y = np.split(data, [-1], axis=1)
    x=x[:,1:].astype("int")
    y=y.astype('int')
    # 特征缩放
    mean = np.mean(x, 0)  # 平均数
    sigma = np.std(x, 0, ddof=1)  # 标准差
    x = (x - mean) / sigma  # 标准化特征缩放

    # 拼接
    m = len(x)
    x = np.c_[np.ones((m, 1)), x]
    y = np.c_[y]

    # 切分训练集和测试集
    num = int(m * 0.7)
    trainx, testx = np.split(x, [num])
    trainy, testy = np.split(y, [num])


    # sigmoid函数
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))


    # 模型
    def model(x, theta):
        z = x.dot(theta)  # 两个矩阵相乘
        h = sigmoid(z)    # 用sigmoid函数将连续值映射为0-1之间的概率值
        return h


    # 交叉熵代价
    def cost_function(h, y):
        m = len(h)
        J = -1.0 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return J


    # 梯度下降函数
    def gradsDesc(x, y, alpha=0.001, count_iter=15000, lamda=0.5):
        m, n = x.shape
        theta = np.zeros((n, 1))  # 给定初始参数
        jarr = np.zeros(count_iter)

        for i in range(count_iter):
            h = model(x, theta)  #返回sigmoid函数值
            e = h - y
            jarr[i] = cost_function(h, y) # 计算交叉熵值
            deltatheta = 1.0 / m * x.T.dot(e)  # 计算梯度值
            theta -= alpha * deltatheta   #alpha 为步长，更新参数值

        return jarr, theta


    # 模型精度，准确率
    def accuracy(y, h):
        m = len(y)
        count = 0  # 统计预测值与真实值一致的样本个数
        for i in range(m):
            h[i] = np.where(h[i] >= 0.5, 1, 0)  # 将预测值从概率值转换为0或1
            if h[i] == y[i]:
                count += 1

        return count / m


    # 画图
    def draw(x, y, theta):
        zeros = y[:, 0] == 0  # 选取y=0的行，其值为true
        ones = y[:, 0] == 1  # 选取y=1的行，其值为true

        # 画散点图
        plt.scatter(x[zeros, 1], x[zeros, 2], c='b', label='负向类')  # 画负向类的散点图
        plt.scatter(x[ones, 1], x[ones, 2], c='r', label='正向类')  # 画正向类的散点图

        # 画分界线
        # 取x1的最小值和最大值
        minx1 = x[:, 1].min()
        maxx1 = x[:, 1].max()

        # 计算x1的最大值和最小值在z=0上的对应的x2值
        minx1_x2 = -((theta[0] + theta[1] * minx1) / theta[2])
        maxx1_x2 = -((theta[0] + theta[1] * maxx1) / theta[2])

        # 以两个点坐标，画出z=0的决策边界
        plt.plot([minx1, maxx1], [minx1_x2, maxx1_x2])
        plt.title('测试精度:%0.2f' % (accuracy(testy, testh)))
        plt.legend()
        plt.show()


    # 训练模型
    jarr, theta = gradsDesc(trainx, trainy)

    # 计算测试值预测值
    testh = model(testx, theta)

    # 计算测试集预测精度
    print('测试集预测精度：', accuracy(testy, testh))
    # print('测试集预测值：', testh)

    # 画图
    draw(x, y, theta)

# 画sigmoid函数
# a = np.arange(-10, 10)
# print(a)
# b = sigmoid(a)
# plt.plot(a,b)
# plt.show()

if __name__ == "__main__":
    fit_model0()
