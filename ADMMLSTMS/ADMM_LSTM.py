# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 10:18
# @Author  : liushuo
# @FileName: ADMM_LSTM.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inspect import Parameter
from typing import Tuple

import tensorflow as tf
import numpy as np
from tensorflow import Tensor
import torch


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values):
    return values*(1-values)

def tanh_derivative(values):
    return 1. - values ** 2

class NaiveLstm:
    def __init__(self, input_size: int, hidden_size: int):
        super(NaiveLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门的权重矩阵和bias矩阵
        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(Tensor(hidden_size, 1))

        # 遗忘门的权重矩阵和bias矩阵
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(Tensor(hidden_size, 1))

        # 输出门的权重矩阵和bias矩阵
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(Tensor(hidden_size, 1))

        # cell的权重矩阵和bias矩阵
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(Tensor(hidden_size, 1))

        self.w_yh = Parameter(Tensor(input_size, hidden_size))
        self.b_yh = Parameter(Tensor(input_size, 1))

    def forward(self, inputs: Tensor, state: Tuple[Tensor]):

        x, i, f, g, o, s, h, y = {}, {}, {}, {}, {}, {}, {}, {}
        h[-1] = torch.zeros(1, self.hidden_size).t()
        s[-1] = torch.zeros(1, self.hidden_size).t()

        seq_size = 24
        for t in range(seq_size):
            x[t] = inputs[:, t, :].t()
            # input gate
            i[t] = torch.sigmoid(self.w_ii @ x[t] + self.w_hi @ h[t-1] + self.b_i)
            # forget gate
            f[t] = torch.sigmoid(self.w_if @ x[t] + self.w_hf @ h[t-1] + self.b_f)
            # cell
            g[t] = torch.tanh(self.w_ig @ x[t] + self.w_hg @ h[t-1] + self.b_g)
            # output gate
            o[t] = torch.sigmoid(self.w_io @ x[t] + self.w_ho @ h[t-1] + self.b_o)

            s[t] = f[t] * s[t-1] + g[t] * i[t]
            h[t] = o[t] * torch.tanh(s[t])
            y[t] = self.w_yh @ h[t] + self.b_yh
        return y

    def lossFun(self, targets, preds):
        loss = np.mean(np.square(preds - targets))
        return loss


