#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/19 5:23 下午 
:@File : data_deal
:Version: v.1.0
:Description:
对原始数据处理
"""

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def null_data(df):
    # 删除空值：
    # df.dropna() # 默认会删除所有的空值，包含所有行里面的、所有列里面的空值
    # df.dropna(how="all") # 删除全为 NA 的行(即如果一整行的单元格都是NA的话，那么则删除这一行)
    # df.dropna(axis=1, how="all") # 删除全为 NA 的列(即如果一整列的单元格都是NA的话，那么则删除这一列)
    #
    # #判断空值：
    # df.isnull() # 对应的返回布尔值，查看是否为空的数据
    # df.notnull() # 与上一条相反

    # replace
    df_re = df.replace(to_replace="", value=np.nan, inplace=True, regex=False)
    # - to_replace：被替换的字符
    # - value：替换后的字符
    # - inplace：是否在原基础上进行修改
    # - regex：是否为正则表达式方式

    # # 填充缺失值：
    # df.fillna(value=2, inplace=True) # 对缺失的NAN值全部替换为 0
    # df.fillna(value=2, method="ffill", limit=2)
    # # - method：表示插值方式，默认为ffill，对这个字段更多详解请参考官网文档
    # - limit：表示可以连续填充的最大次数(对于前向和后向填充)
    # - value：需要填充的值，可以为标量(字符串、数值等)或者字典对象
    # - inspace：表示是否在原来的对象上进行修改而不产生新的对象，true：在原基础上修改；false：会返回处理后的，新的对象
    return df_re


def data_deal_func(path):
    # 1、读取数据并生成x,y的dataframe
    data_df = pd.read_json(path)
    ods_data = data_df["DATA"].str.split(':', expand=True)
    ods_data.columns = ['y', 'x']
    y = ods_data['y'].astype("int").to_frame()
    x = ods_data['x'].str.split('|', expand=True)
    x.columns = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9']
    # 2、缺失值处理
    # null_data(x)
    # x=x.loc[(x["id7"] != "0") & (x["id9"] != '0') & (x['id8'] !='0')] # 按照条件过滤
    # 3、更改数据类型成int
    for i in range(1, 10):
        x['id%s' % i] = pd.to_numeric(x['id%s' % i], errors='coerce').astype(
            float)  # 如果'raise'，则无效解析将引发异常。—如果'coerce'，则无效解析将被设置为NaN。-如果'ignore'，则无效解析将返回输入。
    # 4、均值替换
    for column in list(x.columns[x.isnull().sum() > 0]):
        mean_val = x[column].median()  # 中位数填充
        x[column].fillna(mean_val, inplace=True)

    # 4、数据标准化
    # scaler = MinMaxScaler()  # 这里相当于创建一个归一化的类
    #
    # result = scaler.fit_transform(x_data) # 对data 数据进行归一化
    # print(result) # 输出归一化后的数据
    # print(scaler.inverse_transform(result)) # 可以对归一化后的数据进行逆转

    # 5、数据标准化
    scaler = StandardScaler()  # 然后生成一个标准化对象
    scaler = scaler.fit_transform(x[['id7', 'id8', 'id9']])
    sc_X = pd.DataFrame(data=scaler, columns=['id7', 'id8', 'id9'])

    # 6、重新生成x
    x_end = pd.concat([x[['id1', 'id2', 'id3', 'id4', 'id5', 'id6']], sc_X], axis=1, ignore_index=True)
    train_data, test_data, train_label, test_label = train_test_split(x_end.values, y.values, random_state=1, train_size=0.6,
                                                                      test_size=0.4)

    return train_data, test_data, train_label, test_label


