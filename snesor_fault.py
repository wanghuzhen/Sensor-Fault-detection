# @File    :   snesor_fault.py
# @Version :   2.3
# @Author  :   Wang Huzhen
# @Email   :   2327253081@qq.com
# @Time    :   2020/04/09 09:45:06
import numpy as np
from RNN_model import pre, draw_picture
import random


def avg(list1):
    sum = 0
    for i in list1:
        sum += i
    return (sum * 0.1)/len(list1)


# 最小二乘法计算估计值与实际值之间线性关系
def Least_Square_Method(list_x, list_y):
    avg_x = avg(list_x)
    avg_y = avg(list_y)
    sum_avg_xy = avg_x * avg_y
    sum_xy = 0
    for x, y in zip(list_x, list_y):
        sum_xy += x * y
    sum_x2 = 0
    for x in list_x:
        sum_x2 += x * x
    avg_x2 = avg_x * avg_x
    a = (sum_xy - len(list_x) * sum_avg_xy)/(sum_x2-len(list_x) * avg_x2)
    b = avg_y - a * avg_x
    return a, b


# 输入传感器估计值，实际值和神经网络预测误差
# 判断是否存在故障
def Sensor_Fault_Detection(list_estimate, list_actual, max_dvalue):
    threshold = max_dvalue  # 使用最大残差值作为阈值
    list_est = np.array(list_estimate)
    list_act = np.array(list_actual)
    list_e = list_est - list_act  # 传感器估计值与实际值的残差
    fault = 0  # 初始设置为没有错误
    for i in list_e:
        if abs(i) > max_dvalue:
            fault = 1
            break
    if not fault:
        return 0
    else:
        # 此处实现各类故障判断
        Stuck_fault = 0
        D_value = []
        for k in range(20,len(list_actual)-1):
            D_value.append(list_actual[k]-list_actual[k+1])
            if abs(D_value[k]) >= 0.1:
                # 阈值参考具体参数
                Stuck_fault = 1
        if Stuck_fault:  # 卡死故障/完全故障
            return 1
        a, b = Least_Square_Method(list_estimate, list_actual)
        if abs(a-1) <= 0.01:  # 之所以设置这个大小是忽略最小二乘法拟合时的误差与本身就存在的预测值的误差
            if b > 0.01:
                return 2
            if a-1 > 0:
                return 3
            if a-1 < 0:
                return 4


# 阈值计算
def Threshold(list_estimate, list_actual, snesor_id):
    d_value = list_actual - list_estimate
    threshold = 0
    for i in d_value[:, snesor_id].tolist():
        if threshold < abs(i):
            threshold = abs(i)
    return threshold


# 故障植入，仿真故障，生成故障数据
def fault_insertion(actual_value):
    # 传感器故障插入结果，传感器一:卡死故障/完全故障;传感器二:恒增益故障;传感器三:固定偏差;传感器四:精度下降;
    # 插入卡死故障，从某一时刻开始后面值全固定
    max_value = max(actual_value[:, 0].tolist())  # max和min是为了固定值取在最大值和最小值间
    min_value = min(actual_value[:, 0].tolist())
    actual_value[19:, 0] = random.uniform(min_value, max_value)
    # 插入恒增益或精度下降故障，muti[1, 1]的值大于1为传感器2插入恒增益，muti[3, 3]小于1为传感器4插入精度下降
    muti = np.eye(4, dtype=float)
    muti[1, 1] = 2.
    muti[3, 3] = 0.8
    actual_value = np.matmul(actual_value, muti)
    # 插入固定偏差故障
    bias = np.zeros((49, 4))
    for i in range(bias.shape[0]):
        bias[i][2] = 4.
    actual_value = actual_value+bias
    # print(bias)
    return actual_value


# 输出故障名称
def fault(list_estimate, list_actual, snesor_id):
    max_dvalue = Threshold(list_estimate, list_actual, snesor_id)
    list_actual = fault_insertion(list_actual)  # 插入故障进行识别
    l1 = list_estimate[:, snesor_id].tolist()
    l2 = list_actual[:, snesor_id].tolist()
    fault_str = ''
    fault_type = Sensor_Fault_Detection(
        l1, l2, max_dvalue)  # 故障判断，未输入估计值实际值
    if fault_type == 0:
        fault_str = '无故障'
        # print('无故障')
    elif fault_type == 1:
        fault_str = '完全故障'
        # print('完全故障')
    elif fault_type == 2:
        fault_str = '固定偏差'
        # print('固定偏差')
    elif fault_type == 3:
        fault_str = '恒增益'
        # print('恒增益')
    else:
        fault_str = '精度下降'
        # print('精度下降')
    return fault_str


if __name__ == "__main__":
    result, actual = pre('data&model/sensor_test_1.csv')
    # draw_picture(result, actuall)  # 绘制原始数据图像
    snesor_id = 0
    # l1 = result[:, 0].tolist()
    # l2 = actuall[:, 0].tolist()
    print(fault(result, actual, snesor_id))
