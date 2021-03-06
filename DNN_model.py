# @File    :   RNN_model.py
# @Version :   1.5.3
# @Author  :   Wang Huzhen
# @Email   :   2327253081@qq.com
# @Time    :   2020/04/15 18:00:20
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from sensor_read import read_data, read_testdata
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)


# 获取数据集合并归一化处理
def get_data(data_path=''):
    scaler = StandardScaler()
    if data_path is '':
        (X_train, y_train), (X_test, y_test) = read_data()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, y_train, X_test_scaled, y_test
    else:
        x_test, y_test = read_testdata(data_path)
        x_test_scaled = scaler.fit_transform(x_test)
        return x_test_scaled, y_test


# 创建模型
def create_model(X_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(9, input_shape=X_shape, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.AlphaDropout(rate=0.3))
    model.add(keras.layers.Dense(13, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(13, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.AlphaDropout(rate=0.3))
    model.add(keras.layers.Dense(13, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(13,activation = 'relu'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(13,activation = 'relu'))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    # print(model.summary())
    return model


# 创建模型，编译模型，保存模型,训练模型
# 保存模型可以只保存权重和偏置，可以使用callbacks自动保存模型
# callbacks = [keras.callbacks.ModelCheckpoint(
#     filepath,
#     monitor='val_loss',
#     save_weights_only=True,
#     verbose=1,
#     save_best_only=True,
#     period=1)]
# 下面函数没有使用上面的方法
def model_train(epoch, X_train_scaled, y_train, X_test_scaled, y_test):
    # 获取训练集shape
    X_shape = X_train_scaled.shape[1:]
    # print(X_shape)
    if os.path.exists('data&model/Dnn_model.h5'):
        restored_model = tf.keras.models.load_model('data&model/Dnn_model.h5')
        history = restored_model.fit(X_train_scaled, y_train, validation_data=(
            X_test_scaled, y_test), epochs=epoch)
        # 保存训练模型的权重和偏置
        restored_model.save('data&model/Dnn_model.h5')
        # 删除模型
        del restored_model
    else:
        model = create_model(X_shape)
        # lr为学习率，上次设置为0.01
        model.compile(optimizer=tf.keras.optimizers.Adam(
            lr=0.001), loss='mse', metrics=["accuracy"])
        history = model.fit(X_train_scaled, y_train, validation_data=(
            X_test_scaled, y_test), epochs=epoch)
        # 保存训练模型的权重和偏置
        model.save('data&model/Dnn_model.h5')
        # 删除模型
        del model
    return history


# 使用的数据集是原数据集分割出来的
# 训练后的模型进行预测和评估，返回值是预测值和实际值的差值
def train_predict_evalute():
    X_train_scaled, y_train, X_test_scaled, y_test = get_data()
    # history = model_train(5, X_train_scaled, y_train, X_test_scaled, y_test)
    model = tf.keras.models.load_model('data&model/Dnn_model.h5')
    l1 = np.array(model.predict(X_test_scaled))
    l2 = np.array(y_test)
    del model
    return (l1-l2)


# 预测测试集，使用的数据集是正式有故障的数据集
def pre_DNN(data_path):
    x_test_scaled, y_test = get_data(data_path)
    model = tf.keras.models.load_model('data&model/Dnn_model.h5')
    l1 = np.array(model.predict(x_test_scaled))
    l2 = np.array(y_test)
    del model
    return l1, l2


# 显示数据图像
def draw_picture_DNN(res, act, sensor_type):
    for i in range(4):
        res = res[:, i].tolist()
        act = act[:, i].tolist()
        plt.figure('Sensor'+str(i+1))
        plt.title('Sensor'+str(i+1)+sensor_type['Sensor'+str(i+1)])
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(res, label='估计值--平均值：'+str(round(sum(res)/len(res), 4)))
        plt.plot(act, label='实际值--平均值：'+str(round(sum(act)/len(act), 4)))
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('angel_rate')
        plt.savefig('Sensor'+str(i+1)+'.png')
    plt.show()


if __name__ == '__main__':
    # result = train_predict_evalute()
    result, actuall = pre_DNN('data&model/sensor_test_1.csv')
    print(result.tolist())
    print('======================')
    print(actuall.tolist())
    # plt.plot(result[:, 0].tolist())
    # plt.show()
