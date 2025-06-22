# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 10:19
# @Author  : liushuo
# @FileName: Load_data.py
# @Software: PyCharm
import numpy as np
import xlrd
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



# ----------数据预处理/GEFCom2012数据集----------
def pre_data():
    # 读取数据集/本实验只读取负荷数据，用前24个小时的负荷预测第25时刻的负荷
    Load_data = xlrd.open_workbook("F:/Liushuo/ADMM_Load/dataset/GEFCom2012负荷预测数据集/Load Forecasting/Load2004_2.0.xls")
    # Load_data = xlrd.open_workbook("D:/001/001科研/Scientific Research/RECENTLY/Data/Dataset/GEFCom2012负荷预测数据集/Load Forecasting/Load2004_2.0.xls")
    # 获取一个工作表，通过名称获取/sheet1表示第一个工作表
    load_data = Load_data.sheet_by_name('Sheet1')
    # temp_data = Temp_data.sheet_by_name('Sheet1')
    Load = []
    # Temp = []
    # 将数据存储在二维数组
    for i in range(0, 366):
        data1 = []
        # data2 = []
        for j in range(0, 24):
            data1.append(load_data.cell_value(i, j))
            # data2.append(temp_data.cell_value(i, j))
        Load.append(list(data1))
        # Temp.append(list(data2))
    # 将数据存储在列表/从左到右，从上到下
    l = torch.zeros((24*366,))
    l_eq = torch.zeros((24*366,))
    # t = np.zeros((24*366,))
    for i in range(366):
        for j in range(24):
            l[i * 24 + j] = Load[i][j]
            # t[i * 24 + j] = Temp[i][j]
    # 归一化
    L_max = max(l)
    # T_max = max(t)
    L = list()
    # T = list()
    for i in range(0, 24 * 366):
        value_l = l[i] / L_max
        # value_t = t[i]/ T_max
        L.append(value_l)
        # T.append(value_t)
    # 划分训练集、验证集和测试集/7:2:1
    x_L_train = []
    # x_T_train = []
    y_train = []
    x_L_val = []
    y_val = []
    x_L_test = []
    # x_T_test = []
    y_test = []
    for i in range(24, 7896):
        x_L_train.append(L[i - 24:i])
        # x_T_train.append(T[i - 24:i])
        y_train.append(L[i])
    # for i in range(6168, 7896):
    #     x_L_val.append(L[i - 24:i])
    #     # x_T_test.append(T[i - 24:i])
    #     y_val.append(L[i])
    for i in range(7920, 8784):
        x_L_test.append(L[i - 24:i])
        # x_T_test.append(T[i - 24:i])
        y_test.append(L[i])
    x_train = torch.tensor(x_L_train).to(device)
    y_train = torch.tensor(y_train).to(device)
    y_train = y_train.reshape(7872, 1)

    # x_val = torch.tensor(x_L_val).to(device)
    # y_val = torch.tensor(y_val).to(device)
    # y_val = y_val.reshape(1728, 1)
    
    x_test = torch.tensor(x_L_test).to(device)
    y_test = torch.tensor(y_test).to(device)
    y_test = y_test.reshape(864, 1)
    # y_test = y_test * L_max
    #
    # f = open("D:/Liushuo/Exp-1/data/Actual_load.txt", 'a')
    # for i in range(len(y_test)):
    #     f.write(str(y_test[i].item()*L_max.item()) + '\n')
    # f.close()
    
#     x_train = np.array(x_L_train) #(2,6120,24)
#     y_train = np.array(y_train).reshape(24360, 1)
#     x_test = np.array(x_L_test) #(2,864,24)
#     y_test = np.array(y_test).reshape(17520, 1)

    return x_train, y_train, x_test, y_test, L_max

x_train, y_train, x_test, y_test, L_max = pre_data()

# print(x_train)
