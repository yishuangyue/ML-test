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
import xlsxwriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Logistic.data_deal import data_deal
import joblib


# def data_deal(path):
#     # 1.读取数据集
#     data = np.loadtxt(path, dtype=str, delimiter=",")
#     # 2.划分数据与标签
#     x, y = np.split(data, indices_or_sections=(5,), axis=1)  # x为数据，y为标签
#     train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6,
#                                                                       test_size=0.4)
#     return train_data, test_data, train_label, test_label


def fit_svm(train_data, test_data, train_label, test_label):
    # print(train_data.shape)
    # 3.训练svm分类器
    classifier = svm.SVC(C=2, kernel='rbf', gamma=8, decision_function_shape='ovo')  # rbf:高斯核函数
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

    # 3.1模型保存与加载
    joblib.dump(classifier, model_path)  # 保存模型
    # model1 = joblib.load(model_output)  # 加载模型
    # print("加载模型完毕\n")

    # 4.计算svc分类器的准确率
    print("训练集准确率：", classifier.score(train_data, train_label))
    print("测试集精度：", classifier.score(test_data, test_label))

    # or 也可直接mi调用accuracy_score方法计算准确率
    tra_label = classifier.predict(train_data)  # 训练集的预测标签
    tes_label = classifier.predict(test_data)  # 测试集的预测标签
    print("训练集准确率：", accuracy_score(train_label, tra_label))
    print("测试集精度：", accuracy_score(test_label, tes_label))
    # 查看决策函数
    # print('train_decis 1ion_function:\n',classifier.decision_function(train_data)) # (90,3)
    # print('predict_result:\n', classifier.predict(train_data))
    return tra_label, tes_label


# 5 .写入预测值########
def write_predict(x_pred, real, pred, output_path):
    global j
    workbook = xlsxwriter.Workbook(str(output_path))  # 保存地址
    worksheet = workbook.add_worksheet('Sheet1')
    format4 = workbook.add_format(
        {'font_size': '12', 'align': 'center', 'valign': 'vcenter', 'bold': True, 'font_color': '#217346',
         'bg_color': '#FFD1A4'})
    col = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1']
    # 　　设置自己想要的标题
    title = ['djxh', 'zlxqwgz_bz', 'tfrztfzc_bz', 'lx3ywsb_bz', 'sfz_bz', 'nsrzt_bz', "pred"]
    worksheet.write_row(col[0], title, format4)  # 设置AF-AI列的样式
    for i in range(len(pred)):
        for j in range(len(x_pred[0])):
            worksheet.write(i + 1, j, x_pred[i][j])  # 写入预测样本
        worksheet.write(i + 1, j + 1, real[i])  # 写入预测样本真实值
        worksheet.write(i + 1, j + 2, pred[i])  # 写入预测值
    workbook.close()
    print('数据写入完成')


# main入口

if __name__ == '__main__':
    # input_path = '/Users/liting/Documents/python/Moudle/ML-test/svm/formatData.txt'
    output_path = '/Users/liting/Documents/python/Moudle/ML-test/svm/Results.xlsx'
    model_path = '/Users/liting/Documents/python/Moudle/ML-test/svm/train1_model.m'
    # train_data, test_data, train_label, test_label = data_deal(input_path)
    # train_data = train_data[:, 1:]
    # test_data = test_data[:, 1:]

    input_path = '/Users/liting/Documents/python/Moudle/ML-test/Logistic/ods_data.json'
    train_data, test_data, train_label, test_label = data_deal(input_path)
    tra_label, tes_label = fit_svm(train_data, test_data, train_label, test_label)
    # write_predict(test_data, test_label, tes_label, output_path)
