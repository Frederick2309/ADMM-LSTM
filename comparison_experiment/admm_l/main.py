# -*- coding: utf-8 -*-
# @Time    : 2023/1/6 9:47
# @Author  : liushuo
# @FileName: main.py
# @Software: PyCharm
import os
from _global import info
from demo import save_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import comparison_experiment.admm_l.admm_lstm as admm_lstm
from torch.nn import Parameter
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from pre_data import *
import matplotlib

matplotlib.use("TkAgg")
# warnings.filterwarnings('ignore')
# torch.manual_seed(1)
# 定义超参数
from typing import Dict, List


class LSTM_L(nn.Module):
    def __init__(self, input_size, hidden_size, Ui, Wi, Wo, Uf, Wf, Uo, Ug, Wg, Wy):
        super(LSTM_L, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input gate
        self.W_hi = Parameter(Ui)
        self.W_ii = Parameter(Wi)
        # forget gate
        self.W_hf = Parameter(Uf)
        self.W_if = Parameter(Wf)
        # output gate
        self.W_ho = Parameter(Uo)
        self.W_io = Parameter(Wo)
        # cell
        self.W_hg = Parameter(Ug)
        self.W_ig = Parameter(Wg)
        # linear
        self.W_y = Parameter(Wy)

    def forward(self, x, init_states=None):
        batch_size, seq_size, _ = x.size()
        if init_states is None:
            h_t = torch.zeros((batch_size, self.hidden_size)).to(device)
            c_t = torch.zeros((batch_size, self.hidden_size)).to(device)
        else:
            h_t, c_t = init_states
        for _t in range(x.size(1)):
            x_t = x[:, _t, :]
            zi_t = torch.matmul(h_t, self.W_hi).to(device) + torch.matmul(x_t, self.W_ii).to(device)
            i_t = torch.sigmoid(zi_t).to(device)
            zf_t = torch.matmul(h_t, self.W_hf).to(device) + torch.matmul(x_t, self.W_if).to(device)
            f_t = torch.sigmoid(zf_t).to(device)
            zo_t = torch.matmul(h_t, self.W_ho).to(device) + torch.matmul(x_t, self.W_io).to(device)
            o_t = torch.sigmoid(zo_t).to(device)
            zg_t = torch.matmul(h_t, self.W_hg).to(device) + torch.matmul(x_t, self.W_ig).to(device)
            g_t = torch.tanh(zg_t).to(device)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t).to(device)
        output = torch.matmul(h_t, self.W_y).to(device)
        return output


def admm_l_demo(num_epochs, n_hiddens, train_x, train_y, test_x, test_y, save=False) -> Dict[str, List[float] or str]:
    n_batches = train_x.size(0)
    seq_length = train_x.size(1)
    n_feature = test_x.size(2)
    Wf = torch.randn(n_feature, n_hiddens).to(device) * 0.1
    Uf = torch.randn(n_hiddens, n_hiddens).to(device) * 0.1
    Wi = torch.randn(n_feature, n_hiddens).to(device) * 0.1
    Ui = torch.randn(n_hiddens, n_hiddens).to(device) * 0.1
    Wo = torch.randn(n_feature, n_hiddens).to(device) * 0.1
    Uo = torch.randn(n_hiddens, n_hiddens).to(device) * 0.1
    Wg = torch.randn(n_feature, n_hiddens).to(device) * 0.1
    Ug = torch.randn(n_hiddens, n_hiddens).to(device) * 0.1
    Wy = torch.randn(n_hiddens, 1).to(device) * 0.1

    def LSTM_Forward(_input):
        x, zf, f, zi, i, zo, o, zg, g, c, h, pred = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        h[-1] = torch.zeros((_input.shape[0], n_hiddens)).to(device)
        c[-1] = torch.zeros((_input.shape[0], n_hiddens)).to(device)
        for t in range(seq_length):
            x[t] = _input[:, t, :]
            zf[t] = torch.matmul(x[t], Wf).to(device) + torch.matmul(h[t - 1], Uf).to(device)  #(n_batches,n_hiddens)
            f[t] = torch.sigmoid(zf[t]).to(device)
            zi[t] = torch.matmul(x[t], Wi).to(device) + torch.matmul(h[t - 1], Ui).to(device)
            i[t] = torch.sigmoid(zi[t]).to(device)
            zo[t] = torch.matmul(x[t], Wo).to(device) + torch.matmul(h[t - 1], Uo).to(device)
            o[t] = torch.sigmoid(zo[t]).to(device)
            zg[t] = torch.matmul(x[t], Wg).to(device) + torch.matmul(h[t - 1], Ug).to(device)
            g[t] = torch.tanh(zg[t]).to(device)
            c[t] = f[t] * c[t - 1] + i[t] * g[t]
            h[t] = o[t] * torch.tanh(c[t]).to(device)
        output = torch.matmul(h[seq_length - 1], Wy).to(device)
        return zf, f, zi, i, zo, o, zg, g, c, h, output

    zf, f, zi, i, zo, o, zg, g, c, h, a = LSTM_Forward(train_x)
    loss_train = []
    loss_test = []

    ##############################################################################
    '''
    初始化超参数z
    '''
    lambda00 = 1e-6  # 1
    lambda02 = 1e-6
    lambda03 = 1e-6  # 0.1
    RHO_singular = 1  # rho1,3,57 #1
    lambda1 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    lambda2 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    lambda3 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    lambda4 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    RHO_plural = 1  # rho2,3,6,8 #1
    lambda5 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    lambda6 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    lambda7 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    lambda8 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    rho9 = 1  #
    lambda9 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    rho10 = 1
    lambda10 = torch.zeros((n_batches, seq_length, n_hiddens)).to(device)
    rho11 = 0.0001  # 0.01
    lambda11 = torch.zeros(n_batches, 1).to(device)
    ##############################################################################

    start_loss = torch.mean(torch.square(train_y - a).to(device)).to(device)
    start_loss_ = torch.mean(torch.square(test_y - LSTM_Forward(test_x)[-1]).to(device)).to(device)
    info(f'Loss at the beginning: {start_loss.item()}')
    loss_train.append(start_loss.item())
    loss_test.append(start_loss_.item())

    for k in range(num_epochs):
        Wy = admm_lstm.update_Wy(a, Wy, h[seq_length - 1], lambda03, lambda11, rho11)
        Wg = admm_lstm.update_W(zg, Wg, train_x, Ug, h, lambda00, lambda7, RHO_singular, seq_length)
        Ug = admm_lstm.update_U(zg, Wg, train_x, Ug, h, lambda02, lambda7, RHO_singular, seq_length)
        Wo = admm_lstm.update_W(zo, Wo, train_x, Uo, h, lambda00, lambda5, RHO_singular, seq_length)
        Uo = admm_lstm.update_U(zo, Wo, train_x, Uo, h, lambda02, lambda5, RHO_singular, seq_length)
        Wi = admm_lstm.update_W(zi, Wi, train_x, Ui, h, lambda00, lambda3, RHO_singular, seq_length)
        Ui = admm_lstm.update_U(zi, Wi, train_x, Ui, h, lambda02, lambda3, RHO_singular, seq_length)
        Wf = admm_lstm.update_W(zf, Wf, train_x, Uf, h, lambda00, lambda1, RHO_singular, seq_length)
        Uf = admm_lstm.update_U(zf, Wf, train_x, Uf, h, lambda02, lambda1, RHO_singular, seq_length)
        for t in range(seq_length):
            zf[t] = admm_lstm.update_z(zf[t], f[t], Wf, train_x[:, t, :], Uf, h[t - 1], lambda1[:, t, :], RHO_singular,
                                       lambda2[:, t, :], RHO_plural)
            f[t] = admm_lstm.update_f(zf[t], g[t], i[t], c[t], c[t - 1], lambda2[:, t, :], RHO_plural, lambda9[:, t, :],
                                      rho9)
            zi[t] = admm_lstm.update_z(zi[t], i[t], Wi, train_x[:, t, :], Ui, h[t - 1], lambda3[:, t, :], RHO_singular,
                                       lambda4[:, t, :], RHO_plural)
            i[t] = admm_lstm.update_i(zi[t], g[t], f[t], c[t], c[t - 1], lambda4[:, t, :], RHO_plural, lambda9[:, t, :],
                                      rho9)
            zo[t] = admm_lstm.update_z(zo[t], o[t], Wo, train_x[:, t, :], Uo, h[t - 1], lambda5[:, t, :], RHO_singular,
                                       lambda6[:, t, :], RHO_plural)
            o[t] = admm_lstm.update_o(zo[t], c[t], h[t], lambda6[:, t, :], RHO_plural, lambda10[:, t, :], rho10)
            zg[t] = admm_lstm.update_zg(zg[t], g[t], Wg, train_x[:, t, :], Ug, h[t - 1], lambda7[:, t, :], RHO_singular,
                                        lambda8[:, t, :], RHO_plural)
            g[t] = admm_lstm.update_g(zg[t], i[t], f[t], c[t], c[t - 1], lambda8[:, t, :], RHO_plural, lambda9[:, t, :],
                                      rho9)
            #
            c[t] = admm_lstm.update_c(f[t], i[t], o[t], g[t], c[t], c[t - 1], h[t], lambda9[:, t, :], rho9,
                                      lambda10[:, t, :], rho10)

            h[t] = admm_lstm.update_h(c[t], o[t], a, Wy, h[t], lambda10[:, t, :], rho10, lambda11, rho11, t,
                                      seq_length)
            if t == seq_length - 1:
                a = admm_lstm.update_a(a, train_y, Wy, h[seq_length - 1], lambda11, rho11)
                lambda11 = admm_lstm.update_lambda11(a, Wy, h[seq_length - 1], lambda11, rho11)

            lambda10[:, t, :] = admm_lstm.update_lambda10(c[t], o[t], h[t], lambda10[:, t, :], rho10, t, seq_length)
            lambda9[:, t, :] = admm_lstm.update_lambda9(c[t], c[t - 1], g[t], i[t], f[t], lambda9[:, t, :], rho9)
            lambda8[:, t, :] = admm_lstm.update_lambda8(zg[t], g[t], lambda8[:, t, :], RHO_plural)
            lambda7[:, t, :] = admm_lstm.update_lambda_singular(zg[t], Wg, train_x[:, t, :], Ug, h[t - 1],
                                                                lambda7[:, t, :], RHO_singular)
            lambda6[:, t, :] = admm_lstm.update_lambda_plural(zo[t], o[t], lambda6[:, t, :], RHO_plural)
            lambda5[:, t, :] = admm_lstm.update_lambda_singular(zo[t], Wo, train_x[:, t, :], Uo, h[t - 1],
                                                                lambda5[:, t, :],
                                                                RHO_singular)
            lambda4[:, t, :] = admm_lstm.update_lambda_plural(zi[t], i[t], lambda4[:, t, :], RHO_plural)
            lambda3[:, t, :] = admm_lstm.update_lambda_singular(zi[t], Wi, train_x[:, t, :], Ui, h[t - 1],
                                                                lambda3[:, t, :],
                                                                RHO_singular)
            lambda2[:, t, :] = admm_lstm.update_lambda_plural(zf[t], f[t], lambda2[:, t, :], RHO_plural)
            lambda1[:, t, :] = admm_lstm.update_lambda_singular(zf[t], Wf, train_x[:, t, :], Uf, h[t - 1],
                                                                lambda1[:, t, :],
                                                                RHO_singular)

        zf1, f1, zi1, i1, zo1, o1, zg1, g1, c1, h1, a1 = LSTM_Forward(train_x)
        zf2, f2, zi2, i2, zo2, o2, zg2, g2, c2, h2, a2 = LSTM_Forward(test_x)
        loss_train.append(torch.mean(torch.square(train_y - a1)).item())
        loss_test.append(torch.mean(torch.square(test_y - a2)).item())

        info(f'ADMM-LSTM-L: k = {k + 1}, loss train = {loss_train[-1]}, loss test = {loss_test[-1]}')

    if save:
        model = LSTM_L(n_feature, n_hiddens, Ui, Wi, Wo, Uf, Wf, Uo, Ug, Wg, Wy)
        save_model('ADMM-LSTM-L', model)

    return {
        'name': 'ADMM-LSTM-L',
        'train_loss': loss_train,
        'val_loss': loss_test,
    }
