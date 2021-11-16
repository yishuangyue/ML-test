#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/10 2:52 下午 
:@File : svm
:Version: v.1.0
:Description:
"""
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import joblib

# define converts(字典)
# def Iris_label(s):
#     it={b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2 }
#     return it[s]
# 1.读取数据集
path = '/Users/liting/Documents/python/Moudle/svm/test.txt'
data = np.loadtxt(path, dtype=str, delimiter=",")
# converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
# print(data.shape)

# 2.划分数据与标签
test_data, test_label= np.split(data, indices_or_sections=(5,), axis=1)  # x为数据，y为标签
test_data = test_data[:, 1:]

# 3.训练svm分类器
# classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # rbf:高斯核函数
# classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

# 3.1模型保存与加载
# joblib.dump(classifier, "/Users/liting/Documents/python/Moudle/svm/train1_model.m")  # 保存模型
model1 = joblib.load("/Users/liting/Documents/python/Moudle/svm/train1_model.m")  # 加载模型
print("加载模型完毕\n")



# 4.计算svc分类器的准确率
print("测试集：", model1.score(test_data, test_label))

# 也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score

tes_label = model1.predict(test_data)  # 测试集的预测标签
print("测试集：", accuracy_score(test_label, tes_label))

# 查看决策函数
print('train_decision_function:\n',model1.decision_function(test_data)) # (90,3)
print('predict_result:\n',model1.predict(test_data))

# 5.绘制图形
# # 确定坐标轴范围
# x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
# x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
# x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
# grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# # 指定默认字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# # 设置颜色
# cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
# cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
#
# grid_hat = classifier.predict(grid_test)  # 预测分类值
# grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
#
# plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=cm_dark)  # 样本
# plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, edgecolors='k', zorder=2,
#             cmap=cm_dark)  # 圈中测试集样本点
# plt.xlabel('花萼长度', fontsize=13)
# plt.ylabel('花萼宽度', fontsize=13)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# print("eeeeeeeend")
# plt.title('SVM特征分类')
# plt.show()
