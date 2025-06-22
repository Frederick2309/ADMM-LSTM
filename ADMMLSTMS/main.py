# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 10:19
# @Author  : liushuo
# @FileName: main.py
# @Software: PyCharm
import numpy as np
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from demo import init
import ADMMLSTMS.common as common
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Parameter
import time
from _global import global_dict

torch.manual_seed(0)

(
    num_epochs, hidden_size,
    ((x_train, y_train, x_test, y_test), example_dict, dataset_name),
    _, save, g_args
) = init('comp')

X_train = x_train
# X_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
X_test = x_test

n_hiddens = hidden_size
n_feature = X_train.shape[2]
n_batches = X_train.shape[0]
seq_length = X_train.shape[1]
# for seed in range(100):
#     torch.manual_seed(seed)  # 3
#     Wf = torch.randn(n_hiddens, n_hiddens).to(device) * 1
#     Uf = torch.randn(n_feature, n_hiddens).to(device) * 1
#     bf = torch.randn(n_hiddens).to(device) * 1
#     Wi = torch.randn(n_hiddens, n_hiddens).to(device) * 1
#     Ui = torch.randn(n_feature, n_hiddens).to(device) * 1
#     bi = torch.randn(n_hiddens).to(device) * 1
#     Wo = torch.randn(n_hiddens, n_hiddens).to(device) * 1
#     Uo = torch.randn(n_feature, n_hiddens).to(device) * 1
#     bo = torch.randn(n_hiddens).to(device) * 1
#     Wc = torch.randn(n_hiddens, n_hiddens).to(device) * 1
#     Uc = torch.randn(n_feature, n_hiddens).to(device) * 1
#     bc = torch.randn(n_hiddens).to(device) * 1
#     Wy = torch.randn(n_hiddens, 1).to(device) * 1
#     by = torch.randn(1).to(device) * 1
#
#
#     x, zf, h, f, zi, i, zo, o, zc, c_, c, y = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#     h[-1] = torch.zeros((x_train.shape[0], n_hiddens)).to(device)
#     c[-1] = torch.zeros((x_train.shape[0], n_hiddens)).to(device)
#         # h[-1] = torch.clone(hprev).to(device)
#         # c[-1] = torch.clone(cprev).to(device)
#     for t in range(seq_length):
#         x[t] = x_train[:, t].reshape(x_train.shape[0], 1)
#         zf[t] = torch.matmul(h[t - 1], Wf).to(device) + torch.matmul(x[t], Uf).to(device) + bf
#         f[t] = common.sigmoid(zf[t])
#         zi[t] = torch.matmul(h[t - 1], Wi).to(device) + torch.matmul(x[t], Ui).to(device) + bi
#         i[t] = common.sigmoid(zi[t])
#         zo[t] = torch.matmul(h[t - 1], Wo).to(device) + torch.matmul(x[t], Uo).to(device) + bo
#         o[t] = common.sigmoid(zo[t])
#         zc[t] = torch.matmul(h[t - 1], Wc).to(device) + torch.matmul(x[t], Uc).to(device) + bc
#         c_[t] = torch.tanh(zc[t]).to(device)
#         c[t] = f[t] * c[t - 1] + i[t] * c_[t]
#         h[t] = o[t] * torch.tanh(c[t]).to(device)
#     y = torch.matmul(h[seq_length - 1], Wy).to(device) + by
#     loss = torch.mean(torch.square(y - y_train).to(device)).to(device)
#
#     A = torch.tensor(0).to(device)
#     B = torch.tensor(0.11).to(device)
#
#     if torch.ge(loss, A) and torch.ge(B, loss):
#         print(seed)
#     if seed == 99:
#         break



# torch.manual_seed(73) #3 #82-0.03 #59 1 3 4 10 18 20 22 38 50 56 73 90 94
torch.manual_seed(0)
Wf = torch.randn(n_hiddens, n_hiddens).to(device)*1
Uf = torch.randn(n_feature, n_hiddens).to(device)*1
bf = torch.randn(n_hiddens).to(device)*1
Wi = torch.randn(n_hiddens, n_hiddens).to(device)*1
Ui = torch.randn(n_feature, n_hiddens).to(device)*1
bi = torch.randn(n_hiddens).to(device)*1
Wo = torch.randn(n_hiddens, n_hiddens).to(device)*1
Uo = torch.randn(n_feature, n_hiddens).to(device)*1
bo = torch.randn(n_hiddens).to(device)*1
Wc = torch.randn(n_hiddens, n_hiddens).to(device)*1
Uc = torch.randn(n_feature, n_hiddens).to(device)*1
bc = torch.randn(n_hiddens).to(device)*1
Wy = torch.randn(n_hiddens, 1).to(device)*1
by = torch.randn(1).to(device)*1


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate
        self.W_hi = Parameter(Wi)
        self.W_ii = Parameter(Ui)
        self.b_i = Parameter(bi)

        # forget gate
        self.W_hf = Parameter(Wf)
        self.W_if = Parameter(Uf)
        self.b_f = Parameter(bf)

        # output gate
        self.W_ho = Parameter(Wo)
        self.W_io = Parameter(Uo)
        self.b_o = Parameter(bo)

        # cell
        self.W_hg = Parameter(Wc)
        self.W_ig = Parameter(Uc)
        self.b_g = Parameter(bc)

        # linear
        self.W_y = Parameter(Wy)
        self.b_y = Parameter(by)

    def forward(self, x, init_states=None):
        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        if init_states == None:
            h_t = torch.zeros((batch_size, self.hidden_size)).to(device)
            c_t = torch.zeros((batch_size, self.hidden_size)).to(device)
        else:
            h_t, c_t = init_states

        for t in range(seq_length):
            x_t = x[:, t, :]
            zi_t = torch.matmul(h_t, self.W_hi).to(device) + torch.matmul(x_t, self.W_ii).to(device) + self.b_i
            i_t = torch.sigmoid(zi_t).to(device)
            zf_t = torch.matmul(h_t, self.W_hf).to(device) + torch.matmul(x_t, self.W_if).to(device) + self.b_f
            f_t = torch.sigmoid(zf_t).to(device)
            zo_t = torch.matmul(h_t, self.W_ho).to(device) + torch.matmul(x_t, self.W_io).to(device) + self.b_o
            o_t = torch.sigmoid(zo_t).to(device)
            zg_t = torch.matmul(h_t, self.W_hg).to(device) + torch.matmul(x_t, self.W_ig).to(device) + self.b_g
            g_t = torch.tanh(zg_t).to(device)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t).to(device)
        output = torch.matmul(h_t, self.W_y).to(device) + self.b_y
        #     hidden_seq.append(h_t)
        # outputs = torch.stack(hidden_seq)
        return output

# LSTM_train()


def lossFun(inputs, targets, hprev, cprev):
    x, zf, h, f, zi, i, zo, o, zc, c_, c, y = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    h[-1] = torch.zeros((inputs.shape[0], n_hiddens)).to(device)
    c[-1] = torch.zeros((inputs.shape[0], n_hiddens)).to(device)
    # h[-1] = torch.clone(hprev).to(device)
    # c[-1] = torch.clone(cprev).to(device)
    for t in range(seq_length):
        x[t] = inputs[:,t].reshape(inputs.shape[0], 1)
        zf[t] = torch.matmul(h[t-1], Wf).to(device) + torch.matmul(x[t], Uf).to(device) + bf
        f[t] = common.sigmoid(zf[t])
        zi[t] = torch.matmul(h[t - 1], Wi).to(device) + torch.matmul(x[t], Ui).to(device) + bi
        i[t] = common.sigmoid(zi[t])
        zo[t] = torch.matmul(h[t - 1], Wo).to(device) + torch.matmul(x[t], Uo).to(device) + bo
        o[t] = common.sigmoid(zo[t])
        zc[t] = torch.matmul(h[t - 1], Wc).to(device) + torch.matmul(x[t], Uc).to(device) + bc
        c_[t] = torch.tanh(zc[t]).to(device)
        c[t] = f[t] * c[t-1] + i[t] * c_[t]
        h[t] = o[t] * torch.tanh(c[t]).to(device)
    y = torch.matmul(h[seq_length - 1], Wy).to(device) + by
    loss =torch.mean(torch.square(y - targets).to(device)).to(device)
    MAPE = torch.mean(torch.abs((y - targets) / targets).to(device)).to(device) * 100
    return zf, h, f, zi, i, zo, o, zc, c_, c, y, loss, MAPE


rho1 = 1
rho2 = 1
rho3 = 1
rho4 = 1
rho5 = 1
rho6 = 1
rho7 = 1
rho8 = 1
rho9 = 1
rho10 = 0.1
rho11 = 0.00001
# rho11 = 0.01
lambda1 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda2 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda3 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda4 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda5 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda6 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda7 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda8 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda9 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda10 = torch.zeros((n_batches, n_hiddens)).to(device)
lambda11 = torch.zeros((n_batches, 1)).to(device)


loss_ = []
loss_val_ = []
loss_test_ = []
ys_ = []
targets_ = []
MAPE_ = []
mu = 0.00000001
alpha = 1
L_max = 1


p = 0
n = 0

# T1 = time.clock()
T1 = time.process_time()
while True:

    if p + n_batches >= 7872 or n == 0:
        hprev = torch.zeros((n_batches, n_hiddens))
        cprev = torch.zeros((n_batches, n_hiddens))
        p = 0

    inputs = x_train
    targets = y_train

    # if n == 0:
    #     zf, h, f, zi, i, zo, o, zc, c_, c, y, loss = lossFun(inputs, targets, hprev, cprev)
    zf, h, f, zi, i, zo, o, zc, c_, c, y, loss0, MAPE0 = lossFun(inputs, targets, hprev, cprev)
    if n == 0:
        loss_.append(loss0)
        # _, _, _, _, _, _, _, _, _, _, pred_val, loss_val, MAPE_val = lossFun(x_val, y_val, hprev, cprev)
        # loss_val_.append(loss_val)
        _, _, _, _, _, _, _, _, _, _, pred_test, loss_test, MAPE_test = lossFun(x_test, y_test, hprev, cprev)
        loss_test_.append(loss_test)
        print("iter %d, loss_train: %f:, loss_test: %f:" % (0, loss0, loss_test))

    N = inputs.shape[1]
    x = {}
    for t in range(N):
        x[t] = torch.zeros((n_batches, 1)).to(device)
        x[t] = inputs[:, t].reshape(n_batches, 1)

    for t in range(N-1, -1, -1):
        if t == N-1:
            y = common.update_y(n_batches, targets, Wy, h, by, lambda11, rho11, t, alpha)
            Wy = common.update_w_yh(y, Wy, h, by, lambda11, rho11, t, alpha)
            by = common.update_b_yh(y, Wy, h, lambda11, rho11, t, alpha)
        h = common.update_h(seq_length, y, Wy, by, h, o, c, x, zf, Wf, Uf, bf, lambda1, rho1, zi, Wi, Ui, bi, lambda3, rho3, zo, Wo, Uo, bo, lambda5, rho5, zc, Wc, Uc, bc, lambda7, rho7, lambda10, rho10, lambda11, rho11, t, alpha)
        o = common.update_o(seq_length, o, c, zo, h, lambda6, rho6, lambda10, rho10, t, alpha)
        zo = common.update_z(seq_length, zo, o, Wo, Uo, h, x, bo, lambda5, rho5, lambda6, rho6, t, alpha)
        Wo = common.update_w(seq_length, zo, Wo, h, Uo, x, bo, lambda5, rho5, mu, alpha)
        Uo = common.update_u(seq_length, zo, Wo, h, Uo, x, bo, lambda5, rho5, mu, alpha)
        bo = common.update_b(seq_length, zo, Wo, h, Uo, x, bo, lambda5, rho5, mu, alpha)
        c = common.update_c(seq_length, c, f, i, c_, o, h, lambda9, rho9, lambda10, rho10, t, alpha)
        f = common.update_f(seq_length, f, c, zf, i, c_, lambda2, rho2, lambda9, rho9, t, alpha)
        zf = common.update_z(seq_length, zf, f, Wf, Uf, h, x, bf, lambda1, rho1, lambda2, rho2, t, alpha)
        Wf = common.update_w(seq_length, zf, Wf, h, Uf, x, bf, lambda1, rho1, mu, alpha)
        Uf = common.update_u(seq_length, zf, Wf, h, Uf, x, bf, lambda1, rho1, mu, alpha)
        bf = common.update_b(seq_length, zf, Wf, h, Uf, x, bf, lambda1, rho1, mu, alpha)
        i = common.update_i(seq_length, i, c_, zi, c, f, lambda4, rho4, lambda9, rho9, t, alpha)
        zi = common.update_z(seq_length, zi, i, Wi, Ui, h, x, bi, lambda3, rho3, lambda4, rho4, t, alpha)
        Wi = common.update_w(seq_length, zi, Wi, h, Ui, x, bi, lambda3, rho3, mu, alpha)
        Ui = common.update_u(seq_length, zi, Wi, h, Ui, x, bi, lambda3, rho3, mu, alpha)
        bi = common.update_b(seq_length, zi, Wi, h, Ui, x, bi, lambda3, rho3, mu, alpha)
        c_ = common.update_c_(seq_length, c_, i, zc, c, f, lambda8, rho8, lambda9, rho9, t, alpha)
        zc = common.update_z_c(seq_length, zc, c_, Wc, Uc, h, x, bc, lambda7, rho7, lambda8, rho8, t, alpha)
        Wc = common.update_w(seq_length, zc, Wc, h, Uc, x, bc, lambda7, rho7, mu, alpha)
        Uc = common.update_u(seq_length, zc, Wc, h, Uc, x, bc, lambda7, rho7, mu, alpha)
        bc = common.update_b(seq_length, zc, Wc, h, Uc, x, bc, lambda7, rho7, mu, alpha)

    lambda1 = common.update_lambda_(seq_length, zf, Wf, h, Uf, x, bf, lambda1, rho1)
    lambda2 = common.update_lambda(seq_length, f, zf, lambda2, rho2)
    lambda3 = common.update_lambda_(seq_length, zi, Wi, h, Ui, x, bi, lambda3, rho3)
    lambda4 = common.update_lambda(seq_length, i, zi, lambda4, rho4)
    lambda5 = common.update_lambda_(seq_length, zo, Wo, h, Uo, x, bo, lambda5, rho5)
    lambda6 = common.update_lambda(seq_length, o, zo, lambda6, rho6)
    lambda7 = common.update_lambda_(seq_length, zc, Wc, h, Uc, x, bc, lambda7, rho7)
    lambda8 = common.update_lambda8(seq_length, c_, zc, lambda8, rho8)
    lambda9 = common.update_lambda9(seq_length, c, f, i, c_, lambda9, rho9)
    lambda10 = common.update_lambda10(seq_length, h, o, c, lambda10, rho10)
    lambda11 = common.update_lambda11(seq_length, y, Wy, h, by, lambda11, rho11)

    zf1, h1, f1, zi1, i1, zo1, o1, zc1, c_1, c1, y1, loss_train, MAPE_train = lossFun(inputs, targets, hprev, cprev)
    # if n>=1:
    #     # loss_.append(loss*L_max)
    loss_.append(loss_train)
    m = n / 1
    # _, _, _, _, _, _, _, _, _, _, pred_val, loss_val, MAPE_val = lossFun(x_val, y_val, hprev, cprev)
    # loss_val_.append(loss_val)
    _, _, _, _, _, _, _, _, _, _, pred_test, loss_test, MAPE_test = lossFun(x_test, y_test, hprev, cprev)
    loss_test_.append(loss_test)
    print("iter %d, loss_train: %f:, loss_test: %f:" % (m + 1, loss_train, loss_test))
    if n == 40:
        T2 = time.process_time()
        Time = (T2 - T1)
        print("iter %d, train_loss: %f:, MAPE_train: %f:" % (m + 1, loss_train, MAPE_train))
        # print("iter %d, val_loss: %f:, MAPE_val: %f:" % (m + 1, loss_val, MAPE_val))
        print("iter %d, test_loss: %f:, MAPE_test: %f:" % (m + 1, loss_test, MAPE_test))
        print("Time: %f:" % (Time))
    if n == 60:
        T2 = time.process_time()
        Time = (T2 - T1)
        print("iter %d, train_loss: %f:, MAPE_train: %f:" % (m + 1, loss_train, MAPE_train))
        # print("iter %d, val_loss: %f:, MAPE_val: %f:" % (m + 1, loss_val, MAPE_val))
        print("iter %d, test_loss: %f:, MAPE_test: %f:" % (m + 1, loss_test, MAPE_test))
        print("Time: %f:" % (Time))
    if n == 80:
        T2 = time.process_time()
        Time = (T2 - T1)
        print("iter %d, train_loss: %f:, MAPE_train: %f:" % (m + 1, loss_train, MAPE_train))
        # print("iter %d, val_loss: %f:, MAPE_val: %f:" % (m + 1, loss_val, MAPE_val))
        print("iter %d, test_loss: %f:, MAPE_test: %f:" % (m + 1, loss_test, MAPE_test))
        print("Time: %f:" % (Time))
    if n == num_epochs - 1:
        T2 = time.process_time()
        Time = (T2 - T1)
        print("iter %d, train_loss: %f:, MAPE_train: %f:" % (m + 1, loss_train, MAPE_train))
        # print("iter %d, val_loss: %f:, MAPE_val: %f:" % (m + 1, loss_val, MAPE_val))
        print("iter %d, test_loss: %f:, MAPE_test: %f:" % (m + 1, loss_test, MAPE_test))
        print("Time: %f:" % (Time))
        # f = open("D:/Liushuo/Exp-1/data/loss_proposed.txt", 'a')
        # for i in range(len(loss_)):
        #     f.write(str(loss_[i].item()) + '\n')
        # f.close()

        # f = open("D:/Liushuo/Exp-1/data/lossval_proposed.txt", 'a')
        # for i in range(len(loss_val_)):
        #     f.write(str(loss_val_[i].item()) + '\n')
        # f.close()

        # f = open("D:/Liushuo/Exp-1/data/losstest_proposed.txt", 'a')
        # for i in range(len(loss_test_)):
        #     f.write(str(loss_test_[i].item()) + '\n')
        # f.close()

        with open("comparison_experiment/admm_s/results.py", 'w') as f:
            f.write('admm_s_loss = { \n'
                    '    "name": "ADMM-LSTM-S", \n'
                    '    "train_loss": [')
            for train_acc in loss_:
                f.write(f'{train_acc.item()}, ')
            f.write('],\n'
                    '    "val_loss": [')
            for val_acc in loss_test_:
                f.write(f'{val_acc.item()}, ')
            f.write(f']\n'
                    '}')

        with open(f"comparison_experiment/admm_s/ADMM-LSTM.{global_dict.get('dataset')}", 'w') as f:
            for train_loss, test_loss in zip(loss_, loss_test_):
                f.write(f'{train_loss:.16f} {test_loss:.16f}\n')

        # plt.plot(loss_, color='green')
        # # plt.plot(loss_val_, color='blue')
        # plt.plot(loss_test_, color='red')
        # plt.xlabel("epoch")
        # plt.ylabel("loss")
        # plt.show()
        break

    p += n_batches  # move data pointer
    n += 1
