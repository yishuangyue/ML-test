#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/19 11:04 上午 
:@File : Logistic1
:Version: v.1.0
:Description:
用skealer自带的包计算
"""
# 导入数值计算的基础库
import joblib
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型函数

from data_deal.data_deal import data_deal_func
# from svm import svm
from sklearn.metrics import accuracy_score


def fit_model(x_fearures, y_label, test_data, test_lable):
    # 调用逻辑回归模型
    lr_clf = LogisticRegression(max_iter=400)
    # 用逻辑回归模型拟合构造的数据集
    lr_clf.fit(x_fearures, y_label)  #
    # 3.1模型保存与加载
    joblib.dump(lr_clf, model_path)  # 保存模型
    print("模型训练准确率：%s" % lr_clf.score(x_fearures, y_label))
    tes_label = lr_clf.predict(test_data)  # 测试集的预测标签
    # lr_clf.predict_proba() # 返回每个样本的概率
    print("测试集精度：", accuracy_score(test_lable, tes_label))

    # 查看其对应模型的w（各项的系数）
    print('the weight of Logistic Regression:', lr_clf.coef_)
    # 查看其对应模型的w0(截距)
    print('the intercept(w0) of Logistic Regression:', lr_clf.intercept_)


# # 模型可视化
# def model_show(x_fearures, y_label, model_path):
#     nx, ny = 200, 100
#     x_min, x_max = plt.xlim()
#     y_min, y_max = plt.ylim()
#     x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
#     # 并分别预测这20000个点y=1的概率，并设置绘制轮廓线的位置为0.5（即y=0.5处，绘制等高线高度），并设置线宽为2，
#     # 颜色为蓝色（图中蓝色线即为决策边界），当然我们也可以将将0.5设置为其他的值，更换绘制等高线的位置，
#     # 同时也可以设置一组0~1单调递增的值，绘制多个不同位置的等高线
#     # 也可以理解为，此时我们将0.5设置为阈值，当p>0.5时，y=1；p<0.5时，y=0，蓝色线就是分界线
#     model1 = joblib.load(model_path)  # 加载模型
#     z_proba = model1.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])
#     z_proba = z_proba[:, 1].reshape(x_grid.shape)
#
#     ## 3\# 可视化测试数据
#     plt.figure()
#     # 可视化测试数据1， 并进行相应的文字注释
#     x_fearures_new1 = np.array([[0, -1]])
#     plt.scatter(x_fearures_new1[:, 0], x_fearures_new1[:, 1], s=50, cmap='viridis')
#     plt.annotate(s='New point 1', xy=(0, -1), xytext=(-2, 0), color='blue',
#                  arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3', color='red'))
#
#     # 可视化测试数据2， 并进行相应的文字注释
#     x_fearures_new2 = np.array([[1, 2]])
#     plt.scatter(x_fearures_new2[:, 0], x_fearures_new2[:, 1], s=50, cmap='viridis')
#     plt.annotate(s='New point 2', xy=(1, 2), xytext=(-1.5, 2.5), color='red',
#                  arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3', color='red'))
#
#     # 可视化训练数据
#     plt.scatter(x_fearures[:, 0], x_fearures[:, 1], c=y_label, s=50, cmap='viridis')
#     plt.title('Dataset')
#
#     # 可视化决策边界
#     plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')
#
#     plt.show()


if __name__ == "__main__":
    # 1、构造数据集 这里直接调用svm中的数据
    input_path = '/Users/liting/Documents/python/Moudle/ML-test/data_deal/ods_data.json'
    model_path = '/Users/liting/Documents/python/Moudle/ML-test/Logistic/train1_model.m'
    train_data, test_data, train_label, test_label = data_deal_func(input_path)
    # train_data, test_data, train_label, test_label = svm.data_deal(input_path)
    x_fearures = train_data  # 将字符串转化为数值
    y_label = train_label[:, 0]  # flatmap一下
    test_lable = test_label[:, 0]
    fit_model(x_fearures, y_label, test_data, test_lable)
