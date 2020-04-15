# @File    :   sensor_read.py
# @Version :   1.2.2
# @Author  :   Wang Huzhen
# @Email   :   2327253081@qq.com
# @Time    :   2020/04/02
# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=np.inf)


def read_data():  # 读取训练数据
    # csv_data = pd.read_csv('data&model/sensor_train.csv')
    # csv_data = pd.read_csv('sensor_train.csv')
    # print(csv_data.shape)
    # print(csv_data)
    # X = np.array(csv_data.iloc[:, list(range(7))])
    # Y = np.array(csv_data.iloc[:, list(range(7, 11))])
    sensor_data = np.loadtxt(
        open("data&model/sensor_data.csv"), delimiter=",", skiprows=1)
    X, y = sensor_data[1:, list(
        range(0, 7))], sensor_data[1:, list(range(7, 11))]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, random_state=42)

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = read_data()
    # X_train = np.array(X_train)
    # Y_train = np.array(Y_train)
    print(X_train.shape)
    # print(X_train)
    print(X_test.shape)
    print(X_test)
    # print(y_test.shape)
    # print(type(X_train[1][1]))
