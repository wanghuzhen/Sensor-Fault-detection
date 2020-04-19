# @File    :   snesor_fault.py
# @Version :   2.1
# @Author  :   Wang Huzhen
# @Email   :   2327253081@qq.com
# @Time    :   2020/04/09 09:45:06
import numpy as np


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
def Sensor_Fault_Detection(list_estimate, list_actual, mean_err):
    threshold = mean_err
    list_est = np.array(list_estimate)
    list_act = np.array(list_actual)
    list_e = list_est - list_act  # 传感器估计值与实际值的残差
    fault = False  # 初始设置为没有错误
    for i in list_e:
        if abs(i) >= mean_err:
            fault = 1
            break
    if not fault:
        return 0
    else:
        # 此处未实现各类故障判断
        Stuck_fault = True  # 初始设置为存在卡死故障
        D_value = []
        for k in range(len(list_actual)-1):
            D_value.append(list_actual[k]-list_actual[k+1])
            if abs(D_value[k]) >= 0.1:
                # 阈值参考具体参数
                Stuck_fault = False
        if Stuck_fault:  # 卡死故障/完全故障
            return 1
        a, b = Least_Square_Method(list_estimate, list_actual)
        if abs(a-1) <= 0.001:
            if b > 0:
                return 2
            if a-1 > 0:
                return 3
            if a-1 < 0:
                return 4


def fault():
    fault_str = ''
    fault_type = Sensor_Fault_Detection()  # 故障判断，未输入估计值实际值
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
    print(fault())
