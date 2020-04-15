# @File    :   RNN_model.py
# @Version :   1.5
# @Author  :   Wang Huzhen
# @Email   :   2327253081@qq.com
# @Time    :   2020/04/15 18:00:20
import tensorflow as tf
from tensorflow import keras
from sensor_read import read_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


# 获取数据集合并归一化处理
def get_data():
    (X_train, y_train), (X_test, y_test) = read_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test


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
    if os.path.exists('data&model/Rnn_model.h5'):
        restored_model = tf.keras.models.load_model('data&model/Rnn_model.h5')
        history = restored_model.fit(X_train_scaled, y_train, validation_data=(
            X_test_scaled, y_test), epochs=epoch)
        # 保存训练模型的权重和偏置
        restored_model.save('data&model/Rnn_model.h5')
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
        model.save('data&model/Rnn_model.h5')
        # 删除模型
        del model
    return history


if __name__ == '__main__':
    X_train_scaled, y_train, X_test_scaled, y_test = get_data()
    history = model_train(5, X_train_scaled, y_train, X_test_scaled, y_test)
    model = tf.keras.models.load_model('data&model/Rnn_model.h5')
    l1 = np.array(model.predict(X_test_scaled))
    l2 = np.array(y_test)
    print(l1-l2)
    # def plot_learning_curver(history):
    #     pd.DataFrame(history.history).plot(figsize = (10, 8))
    #     plt.grid(True)
    #     plt.gca().set_ylim(0, 1)

    # plot_learning_curver(history)
