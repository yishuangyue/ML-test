#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/12/22 4:46 下午 
:@File : rnntest
:Version: v.1.0
:Description:
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# 1、获取训练集测试集
# dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
#                           as_supervised=True)  # 下载好了在这个路径/ /Users/liting/tensorflow_datasets/imdb_reviews/subwords8k/1.0.0.

dataset, metadata = tfds.load(name='imdb_reviews/subwords8k',
                               data_dir='/Users/liting/tensorflow_datasets',
                               download=False, with_info=True,as_supervised=True)



train_dataset, test_dataset = dataset['train'], dataset['test']
# get_next_as_optional
# 2、数据处理
tokenizer = metadata.features['text'].encoder
print('vocabulary size: ', tokenizer.vocab_size)

# # 3 构建批次训练集
BUFFER_SIZE = 10000
BATCH_SIZE = 1000

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))


# 定义模型
# def get_model():
#     inputs = tf.keras.Input((None,))    #  [(None, None)]
#     emb = tf.keras.layers.Embedding(tokenizer.vocab_size, 64)(inputs)   #  (None, None, 64)
#     h1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(emb)     # (None, 128)
#     h1 = tf.keras.layers.Dense(64, activation='relu')(h1)       # (None, 64)
#     outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h1)   # (None, 1)
#     model = tf.keras.Model(inputs, outputs)
#     return model


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64),   # (None, None, 64)
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), # 双向包装器 (None, 128)
        tf.keras.layers.Dense(64, activation='relu'),  # (None, 64)
        tf.keras.layers.Dense(1, activation='sigmoid')  #  (None, 1)
    ])
    return model





strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
with strategy.scope():
        model = get_model()
# 设置训练参数
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

# 模型训练
history = model.fit(train_dataset, epochs=3,
                    validation_data=test_dataset)


# 查看训练过程
# 查看训练过程
#
# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.plot(history.history['val_' + string])
#     plt.xlabel('epochs')
#     plt.ylabel(string)
#     plt.legend([string, 'val_' + string])
#     plt.show()
#
#
# plot_graphs(history, 'accuracy')
# 查看准确度
test_loss, test_acc = model.evaluate(test_dataset)
print('test loss: ', test_loss)
print('test acc: ', test_acc)

# 优化1
# 上述模型不会mask掉序列的padding，所以如果在有padding的寻列上训练，测试没有padding的序列时可能有所偏差。
def pad_to_size(vec, size):
    zeros = [0] * (size-len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad=False):

    tokened_sent = tokenizer.encode(sentence)
    if pad:
        tokened_sent = pad_to_size(tokened_sent, 64)
    pred = model.predict(tf.expand_dims(tokened_sent, 0))
    return pred

# 没有padding的情况
sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

# 有padding的情况
sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print (predictions)





