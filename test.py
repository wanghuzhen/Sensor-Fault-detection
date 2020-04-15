# # 导入相应的库（对数据库进行切分需要用到的库是sklearn.model_selection 中的 train_test_split）
# import time
# from tensorflow import keras
# import tensorflow as tf
# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# # 首先，读取.CSV文件成矩阵的形式。
# my_matrix = np.loadtxt(open("data&model/sensor_data.csv"),
#                        delimiter=",", skiprows=1)
# # 对于矩阵而言，将矩阵倒数第一列之前的数值给了X（输入数据），将矩阵大最后一列的数值给了y（标签）
# X, y = my_matrix[:, 0:7], my_matrix[:, list(range(7, 11))]
# # 利用train_test_split方法，将X,y随机划分问，训练集（X_train），训练集标签（X_test），测试卷（y_train），
# # 测试集标签（y_test），安训练集：测试集=7:3的
# # 概率划分，到此步骤，可以直接对数据进行处理
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, random_state=42)
# # 此步骤，是为了将训练集与数据集的数据分别保存为CSV文件
# # np.column_stack将两个矩阵进行组合连接
# # numpy.savetxt 将txt文件保存为.csv结尾的文件
# train = np.column_stack((X_train, y_train))
# np.savetxt('data&model/sensor_train.csv', train, delimiter=',')
# test = np.column_stack((X_test, y_test))
# np.savetxt('data&model/sensor_test.csv', test, delimiter=',')


# print(np.__version__)
# tf.random.set_seed(22)
# np.random.seed(22)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# assert tf.__version__.startswith('2.')

# total_words = 10000
# max_review_len = 80
# batchsz = 128
# embedding_len = 100

# (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
#     num_words=total_words)
# x_train = keras.preprocessing.sequence.pad_sequences(
#     x_train, maxlen=max_review_len)
# x_test = keras.preprocessing.sequence.pad_sequences(
#     x_test, maxlen=max_review_len)

# db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
# db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# db_test = db_test.shuffle(1000).batch(batchsz, drop_remainder=True)

# print('x_train shape:', x_train.shape,
#       tf.reduce_max(y_train), tf.reduce_min(y_train))
# print('x test shape:', x_test.shape)
import os
check_path = 'data&model/model.ckpt'
check_dir = os.path.dirname(check_path)
print(check_dir)
