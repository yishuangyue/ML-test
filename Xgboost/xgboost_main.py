# -*- coding: utf-8 -*-
import os,joblib,sys
# sys.path.append("/opt/liting/ML-test/")
# sys.path.append("/Users/liting/Documents/python/Moudle/ML-test/")
import operator
import  xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import pyplot, pyplot as plt
from sklearn.model_selection import GridSearchCV,KFold   #通过当前函数去遍历那个参数最好，交叉验证
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from xgboost import plot_importance,XGBClassifier

from data_deal.data_deal import data_deal_func


class XGB():
    def __init__(self, clf_path, vec_path, output_path, if_load):
        '''
        创建对象时完成的初始化工作，判断分类器与vector路径是否为空，
        若为空则创建新的分类器与vector，否则直接加载已经持久化的分类器与vector。
        '''
        self.clf_path = clf_path
        self.vec_path = vec_path
        self.output_path = output_path
        self.if_load = if_load

        if if_load == 0:
            self.vec = TfidfVectorizer()
        else:
            self.vec = joblib.load(self.vec_path)

    # def scoring(self):
    #     #这里的greater_is_better参数决定了自定义的评价指标是越大越好还是越小越好
    #
    #     loss  = make_scorer(logloss, greater_is_better=False)
    #     score = make_scorer(logloss, greater_is_better=True)

    # 训练数据
    # XGBoost结合sklearn网格搜索调参
    def trainXGB(self, X_train, train_lable,X_test,test_lable):
        # 1、train方法训练—
        # param = {'booster':'gbtree',
        #           'max_depth': 5,   # 树最大深度
        #           'eta': 0.01,       # 学习率
        #          'eval_metric': 'error',  # loss计算方式 error二分类
        #          'silent': 0,               # 默认0，值为1时，静默模式开启，不会输出任何信息
        #          'objective': 'binary:logistic',  # 这个参数定义需要被最小化的损失函数，是线性回归，还是二分类'objective': 'reg:logistic' 返回预测概率,多分类softmax直接输出标签，softprob输出概率，返回概率还要多设置num_class：类别数目参数,
        #          }  # 参数 train
        #
        # dtrain=xgb.DMatrix(X_train,train_lable)
        # dttest=xgb.DMatrix(X_test,test_lable)
        # model = xgb.train(params=param
        #                   ,dtrain=dtrain
        #                   ,evals=[(dtrain, 'train'), (dttest, 'valid')]
        #                   ,num_boost_round=1000   # 迭代次数  （决策树数量=迭代次数*类别个数）
        #                   , early_stopping_rounds=10
        #                   )


       # # 2、fit方法训练—
       #  model=XGBClassifier(booster='gbtree',
       #                      learning_rate=0.2,
       #                      n_estimators=50, # 指定树个数
       #                      max_depth=15, # 树的最大深度
       #                      min_child_weight=3,   # 叶子节点最小权重系数
       #                      gamma=0.1,     #叶子节点前面那个系数
       #                      subsample=0.8,    #这是一个常用的使用起始值。典型值介于0.5-0.9之间。对样本选择随机选80%
       #                      colsample_bytree=0.8,  # 这是一个常用的使用起始值。典型值介于0.5-0.9之间选特征的时候随机选择
       #                      objective='binary:logistic',  # 指定损失函数，
       #                      seed=27 ) # 每次复现都是一样的
       #  early_stopping_rounds=10  # 模型连着多少次loss都没有减小（模型没有提升就停下来 10）
       #  eval_metric="error"  #指定的loss值，评估标准 ，回归:rmse - 均方根误差,merror - 多类分类错误率
       #  verbose=True    # 显示时候，每加一个模型，用evals来计算那个loss是怎么样的
       #  model.fit(X_train,train_lable,eval_set=[(X_test,test_lable)],eval_metric=eval_metric,verbose=verbose,early_stopping_rounds=early_stopping_rounds)


        # # 3、指定不同学习率得到输出参数,结合网格搜素调参，XGBoost结合sklearn网格搜索调参
        parameters = {
              'max_depth': [5, 7, 8, 10, 13],
              'learning_rate': [0.02, 0.05, 0.1, 0.2,0.3],
              # 'n_estimators': [30, 40, 50, 80 ],
              # 'min_child_weight': [0, 1, 3, 5, 8],
              # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
              # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
              # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
              # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
              # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
        }
        model = xgb.XGBClassifier(max_depth=5, # 最大深度
                    learning_rate=0.01,   # 学习率
                    n_estimators=30,    #总共迭代的次数，即决策树的个数多给点
                    objective='binary:logistic', # 损失函数
                    eval_metric="error",
                    gamma=0,  # 惩罚项系数，指定节点分裂所需的最小损失函数下降值
                    min_child_weight=1, #  叶子节点最小权重;默认值为1;调参：值越大，越容易欠拟合
                    max_delta_step=0,
                    subsample=0.85,  #  训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典
                    colsample_bytree=0.7,   #  随机选择N%特征建立决策树;防止overfitting
                    colsample_bylevel=1,  # 这是一个常用的使用起始值。典型值介于0.5-0.9之间。对样本选择随机选80%
                    reg_alpha=0,      # 正则化系数
                    reg_lambda=1,     # 正则化系数
                    seed=0)
        grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='neg_log_loss', cv=3,n_jobs=-1)
        # cv交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。
        # scoring:模型评价标准
        # verbose :0：不输出训练过程，1：偶尔输出
        #n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
        #n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
        grid_search.fit(X_train, train_lable)

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set: %s" % grid_search.best_params_ )
        model = grid_search.best_estimator_
        print("grid.best_estimator_:", grid_search.best_estimator_)
        model.fit(X_train,train_lable)


        # # 保存模型(两个)
        model.save_model(self.clf_path)
        # model.dump_model(tree_path)  # 保存树结构
        joblib.dump(self.vec, self.vec_path)
        return model




if __name__ == '__main__':
    # 定义路径以及参数
    # user_path = "/opt/liting/"
    user_path = "/Users/liting/Documents/python/Moudle/"
    data_path = os.path.join(user_path, "ML-test/data_deal/ods_data.json")
    log_path = os.path.join(user_path, "ML_project/data/logs/")
    # 模型保存路径（一个是XGBst模型，一个是TFIDF词进行向量化模型）
    clfmodel_path = os.path.join(user_path,"ML_project/data/model/Xgboost/clf.m")
    tree_path = os.path.join(user_path,"ML_project/data/model/Xgboost/tree")
    vecmodel_path = os.path.join(user_path,"ML_project/data/model/Xgboost/vec.m")
    output_path = os.path.join(user_path,"ML_project/data/model/Xgboost/out.csv")
    # 1、创建NB分类器
    XGB = XGB(vec_path=vecmodel_path, clf_path=clfmodel_path,output_path=output_path, if_load=0)

    # _______创建日志
    # 2、载入训练数据与预测数据
    X_train, X_test, train_label, test_label = data_deal_func(data_path)

     # 训练模型首先需要将分好词的文本进行向量化，这里使用的TFIDF得到词频权重矩阵

    train_lable=train_label[:,0].astype("int")
    test_lable=test_label[:,0].astype("int")
    print("训练集测试集生成好了")
    # 3、训练并预测分类正确性
    trainXGB = XGB.trainXGB(X_train, train_lable,X_test, test_label)
    xgb.to_graphviz(trainXGB, num_trees=1)
    print("模型训练并保存好了")


    #4\预测Predict training set:f
    # train_pred = trainXGB.predict(xgb.DMatrix(X_train)) #
    train_pred = trainXGB.predict(X_train)
    train_pred=pd.DataFrame(train_pred).applymap(lambda x:0 if x<0.5 else 1)
    # pred = trainXGB.predict(X_test) #
    print("训练集准确率：%s" % accuracy_score(train_lable, train_pred))
    # pred = trainXGB.predict(xgb.DMatrix(X_test)) #
    pred = trainXGB.predict(X_test)
    pred=pd.DataFrame(pred).applymap(lambda x:0 if x<0.5 else 1)
    print("测试集准确率：%s" % accuracy_score(test_lable, pred))
    # importance = trainXGB.get_score(importance_type='gain')
    # sorted_importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    print('feature importances[gain]:')
    # print(sorted_importance[0:5])


    # 5、显示重要特征
    # 显示重要特征
    # plot_importance(trainXGB)
    # plt.show()





