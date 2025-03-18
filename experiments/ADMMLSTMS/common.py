# -*- coding: utf-8 -*-
# @Time    : 2022/2/25 16:39
# @Author  : liushuo
# @FileName: common.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from numpy import *
import numpy as np
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def sigmoid(x):
    return 1. / (1 + torch.exp(-x).to(device))


def sigmoid_der(x):
    values = sigmoid(x)
    return values*(1-values)


def tanh_der(x):
    values = torch.tanh(x).to(device)
    return 1. - values ** 2


def update_y(n_batches, target, w_yh, h, b_yh, lambda11, rho11, t, alpha=1):
    temp1 = 2 * target / n_batches + rho11 * torch.matmul(h[t], w_yh).to(device) + rho11 * b_yh - lambda11
    y_old = temp1 / (2 / n_batches + rho11)
    y_new = y_old
    return y_new


def update_w_yh(y, w_yh_old, h, b_yh, lambda11, rho11, t, alpha):
    # r = 0.1  # r #调参
    r = 0.01
    # r = 1
    temp1 = y - torch.matmul(h[t], w_yh_old).to(device) - b_yh + lambda11 / rho11
    temp = rho11 * torch.matmul(torch.t(h[t]).to(device), temp1).to(device)
    res = w_yh_old + temp / r
    w_yh_new = res / 1
    return w_yh_new


def update_b_yh(y, w_yh, h, lambda11, rho11, t, alpha):
    temp1 = y - torch.matmul(h[t], w_yh).to(device) + lambda11 / rho11
    res = torch.mean(temp1, dim = 0).to(device)
    b_yh_new = res / alpha
    return b_yh_new


def Fun(z, h, W, x, U, b, lambda0, rho0, t):
    temp1 = z[t + 1] - torch.matmul(h[t], W).to(device) - torch.matmul(x[t + 1], U).to(device) - b + lambda0/rho0
    temp = torch.matmul(temp1, torch.t(W).to(device)).to(device)
    return temp


def update_h(seq_length, y, w_yh, b_yh, h_old, o, c, x, zf, Wf, Uf, bf, lambda1, rho1, zi, Wi, Ui, bi, lambda3, rho3, zo, Wo, Uo, bo, lambda5, rho5, zc, Wc, Uc, bc, lambda7, rho7, lambda10, rho10, lambda11, rho11, t, alpha):
    r = 100
    mu = 0.00000001
    if t == seq_length - 1:
        temp1 = o[t] * torch.tanh(c[t]).to(device) - lambda10/rho10
        temp2 = y - torch.matmul(h_old[t], w_yh).to(device) - b_yh + lambda11 / rho11
        final = (r - rho10) * h_old[t] + rho10 * temp1 + rho11 * torch.matmul(temp2, torch.t(w_yh).to(device)).to(device)
        h_old[t] = final / r / alpha
    else:
        if t == seq_length - 2:
            temp1 = rho1 * Fun(zf, h_old, Wf, x, Uf, bf, lambda1, rho1, t)
            temp2 = rho3 * Fun(zi, h_old, Wi, x, Ui, bi, lambda3, rho3, t)
            temp3 = rho5 * Fun(zo, h_old, Wo, x, Uo, bo, lambda5, rho5, t)
            temp4 = rho7 * Fun(zc, h_old, Wc, x, Uc, bc, lambda7, rho7, t)
            temp5 = mu * (h_old[t] - o[t] * torch.tanh(c[t])).to(device)
            h_old[t] = (h_old[t] + (temp1 + temp2 + temp3 + temp4 - temp5)/r)/alpha
        else:
            temp1 = Fun(zf, h_old, Wf, x, Uf, bf, 0, 1, t)
            temp2 = Fun(zi, h_old, Wi, x, Ui, bi, 0, 1, t)
            temp3 = Fun(zo, h_old, Wo, x, Uo, bo, 0, 1, t)
            temp4 = Fun(zc, h_old, Wc, x, Uc, bc, 0, 1, t)
            temp5 = h_old[t] - o[t] * torch.tanh(c[t]).to(device)
            h_old[t] = (h_old[t] + (temp1 + temp2 + temp3 + temp4 - temp5) / r) / alpha
    h_new = h_old
    return h_new


def update_o(seq_length, o_old, c, z_o, h, lambda6, rho6, lambda10, rho10, t, alpha):
    if t < seq_length - 1:
        temp1 = sigmoid(z_o[t]) + h[t] * torch.tanh(c[t]).to(device)
        temp2 = 1 / (1 + torch.tanh(c[t]).to(device) * torch.tanh(c[t]).to(device))
        o_old[t] = temp1 * temp2 / alpha
    else:
        temp1 = rho6 * sigmoid(z_o[t]) - lambda6 + rho10 * (h[t] + lambda10/rho10) * torch.tanh(c[t]).to(device)
        temp2 = 1 / (rho6 + rho10 * torch.tanh(c[t]).to(device) * torch.tanh(c[t]).to(device))
        o_old[t] = temp1 * temp2 / alpha
    o_new = o_old
    return o_new


def update_z(seq_length, z_old, output, w, u, h, x, b, lambda_1, rho_1, lambda_2, rho_2, t, alpha):
    if t < seq_length - 1:
        temp_h = 0.5 * (1 + torch.linalg.norm(output[t]).to(device)) + 0.125
        temp1 = torch.matmul(h[t - 1], w).to(device) + torch.matmul(x[t], u).to(device) + b
        temp2 = (sigmoid(z_old[t])-output[t]) * sigmoid_der(z_old[t])
        temp3 = temp1 + 0.5 * temp_h * z_old[t] - temp2
        z_old[t] = 2*temp3 / (2+temp_h) / alpha
    else:
        temp_h = 0.5 * (1 + torch.linalg.norm(output[t] + lambda_2/rho_2).to(device)) + 0.125
        temp1 = torch.matmul(h[t - 1], w).to(device) + torch.matmul(x[t], u).to(device) + b - lambda_1/rho_1
        temp2 = (sigmoid(z_old[t]) - (output[t] + lambda_2/rho_2)) * sigmoid_der(z_old[t])
        temp3 = rho_1 * temp1 + 0.5 * temp_h * z_old[t] - rho_2 * temp2
        z_old[t] = 2 * temp3 / (2*rho_1 + rho_2*temp_h) / alpha
    z_new = z_old
    return z_new


def update_w(seq_length, z, w_old, h, u, x, b, lambda_, rho_, mu, alpha):
    # alpha_ = 1
    temp = torch.zeros_like(w_old).to(device)
    tao = 2400  # Nr
    # mu_ = 0.01
    for t in range(seq_length - 1):
        temp1 = z[t] - torch.matmul(h[t - 1], w_old).to(device) - torch.matmul(x[t], u).to(device) - b
        temp2 = torch.matmul(torch.t(h[t - 1]).to(device), temp1).to(device)
        temp = temp + temp2
    temp1 = z[seq_length - 1] - torch.matmul(h[seq_length - 2], w_old).to(device) - torch.matmul(x[seq_length - 1], u).to(device) - b + lambda_ / rho_
    temp2 = torch.matmul(torch.t(h[seq_length - 2]).to(device), temp1).to(device)
    temp_final = mu * temp + rho_ * temp2
    res = w_old + temp_final / tao
    w_new = res / alpha
    return w_new


def update_u(seq_length, z, w, h, u_old, x, b, lambda_, rho_, mu, alpha):
    # alpha_ = 1
    temp = torch.zeros_like(u_old)
    tao = 2400 # Nr
    for t in range(seq_length - 1):
        temp1 = z[t] - torch.matmul(h[t - 1], w).to(device) - torch.matmul(x[t], u_old).to(device) - b
        temp2 = torch.matmul(torch.t(x[t]).to(device), temp1).to(device)
        temp = temp + temp2
    temp1 = z[seq_length - 1] - torch.matmul(h[seq_length - 2], w).to(device) - torch.matmul(x[seq_length - 1], u_old).to(device) - b + lambda_ / rho_
    temp2 = torch.matmul(torch.t(x[seq_length - 1]).to(device), temp1).to(device)
    temp_final = mu * temp + rho_ * temp2
    res = u_old + temp_final / tao
    u_new = res / alpha
    return u_new


def update_b(seq_length, z, w, h, u, x, b_old, lambda_, rho_, mu, alpha):
    temp = torch.zeros_like(b_old)
    for t in range(seq_length - 1):
        temp1 = z[t] - torch.matmul(h[t - 1], w).to(device) - torch.matmul(x[t], u).to(device)
        temp = temp + temp1
    temp2 = z[seq_length - 1] - torch.matmul(h[seq_length - 2], w).to(device) - torch.matmul(x[seq_length - 1], u).to(device) + lambda_ / rho_
    temp_final = mu * temp + rho_ * temp2
    res = temp_final / ((seq_length - 2) * mu + rho_)
    b_new = torch.mean(res, dim=0).to(device) / alpha
    return b_new


def update_c(seq_length, c_old, f, i, c_, o, h, lambda9, rho9, lambda10, rho10, t, alpha):
    if t < seq_length - 1:
        temp_h = 4 + 2 * torch.linalg.norm(h[t] * 1/o[t]).to(device)
        temp1 = f[t] * c_old[t-1] + i[t] * c_[t]
        temp2 = o[t] * o[t] * temp_h
        temp3 = (o[t] * torch.tanh(c_old[t]) - h[t]).to(device) * o[t] * tanh_der(c_old[t])
        c_old[t] = (2 * temp1 + temp2 * c_old[t] - 2 * temp3)/(2 + temp2)
    else:
        temp_h = 4 + 2 * torch.linalg.norm((h[t] + lambda10/rho10) * 1/o[t]).to(device)
        temp1 = f[t] * c_old[t-1] + i[t] * c_[t] - lambda9/rho9
        temp2 = o[t] * o[t] * temp_h
        temp3 = (o[t] * torch.tanh(c_old[t]).to(device) - (h[t] + lambda10/rho10)) * o[t] * tanh_der(c_old[t])
        c_old[t] = (2 * rho9 * temp1 + rho10 * temp2 * c_old[t] - 2 * rho10 * temp3)/(2 * rho9 + rho10 * temp2)
    c_new = c_old
    return c_new


def update_f(seq_length, f_old, c, z_f, i, c_, lambda2, rho2, lambda9, rho9, t, alpha):
    if t < seq_length - 1:
        temp1 = sigmoid(z_f[t]) + (c[t] - i[t] * c_[t]) * c[t-1]
        temp2 = 1/ (1 + c[t-1] * c[t-1])
        temp3 = temp1 * temp2
        f_old[t] = temp3 / alpha
    else:
        temp1 = rho2 * sigmoid(z_f[t]) - lambda2 + rho9 * c[t-1] * (c[t] - i[t] * c_[t] + lambda9/rho9)
        temp2 = 1 / (rho2 + rho9 * c[t-1] * c[t-1])
        temp3 = temp1 * temp2
        f_old[t] = temp3 / alpha
    f_new = f_old
    return f_new


def update_i(seq_length, i_old, c_, z_i, c, f, lambda4, rho4, lambda9, rho9, t, alpha):
    if t < seq_length - 1:
        temp1 = sigmoid(z_i[t]) + (c[t] - f[t] * c[t-1]) * c_[t]
        temp2 = 1 / (1 + c_[t] * c_[t])
        i_old[t] = (temp1 * temp2) / alpha
    else:
        temp1 = rho4 * sigmoid(z_i[t]) - lambda4 + (rho9 * c[t] - rho9 * f[t] * c[t-1] + lambda9) * c_[t]
        temp2 = 1 / (rho4 + rho9 * c_[t] * c_[t])
        i_old[t] = (temp1 * temp2) / alpha
    i_new = i_old
    return i_new


def update_c_(seq_length, _c_old, i, z_c, c, f, lambda8, rho8, lambda9, rho9, t, alpha):
    if t < seq_length - 1:
        temp1 = torch.tanh(z_c[t]).to(device) + i[t] * (c[t] - f[t] * c[t-1])
        temp2 = 1/(1 + i[t] * i[t])
        _c_old[t] = (temp1 * temp2) / alpha
    else:
        temp1 = rho8 * torch.tanh(z_c[t]).to(device) - lambda8 + i[t] * (rho9 * c[t] - rho9 * f[t] * c[t-1] + lambda9)
        temp2 = 1 / (rho8 + rho9 * i[t] * i[t])
        _c_old[t] = (temp1 * temp2) / alpha
    _c_new = _c_old
    return _c_new


def update_z_c(seq_length, z_c_old, output, w_c, u_c, h, x, b_c, lambda7, rho7, lambda8, rho8, t, alpha):
    if t < seq_length - 1:
        temp_h = 4 + 2 * torch.linalg.norm(output[t]).to(device)
        temp1 = torch.matmul(h[t - 1], w_c).to(device) + torch.matmul(x[t], u_c).to(device) + b_c
        temp2 = (torch.tanh(z_c_old[t]).to(device) - output[t]) * tanh_der(z_c_old[t])
        z_c_old[t] = (2*temp1 + temp_h *z_c_old[t] - 2*temp2) / (2 + temp_h)
    else:
        temp_h = 4 + 2 * torch.linalg.norm(output[t] + lambda8/rho8).to(device)
        temp1 = torch.matmul(h[t - 1], w_c).to(device) + torch.matmul(x[t], u_c).to(device) + b_c - lambda7 / rho7
        temp2 = (torch.tanh(z_c_old[t]).to(device) - (output[t] + lambda8/rho8)) * tanh_der(z_c_old[t])
        temp3 = rho7 * temp1 + 0.5 * temp_h * z_c_old[t] - rho8 * temp2
        z_c_old[t] = 2 * temp3 / (2 * rho7 + rho8 * temp_h) / alpha
    z_c_new = z_c_old
    return z_c_new


def update_lambda(seq_length, output, z, lambda_, rho_): #2,4,6
    temp = output[seq_length - 1] - sigmoid(z[seq_length - 1])
    lambda_new = lambda_ + rho_ * temp
    return lambda_new


def update_lambda_(seq_length, z, w, h, u, x, b, lambda_, rho_): #1,3,5,7
    temp = z[seq_length - 1] - torch.matmul(h[seq_length - 2], w).to(device) - torch.matmul(x[seq_length - 1], u).to(device) - b
    lambda_new = lambda_ + rho_ * temp
    return lambda_new


def update_lambda8(seq_length, c_, z_c, lambda8, rho8):
    temp = c_[seq_length - 1] - torch.tanh(z_c[seq_length - 1]).to(device)
    lambda_new = lambda8 + rho8 * temp
    return lambda_new


def update_lambda9(seq_length, c, f, i, c_, lambda9, rho9):
    temp = c[seq_length - 1] - f[seq_length - 1] * c[seq_length - 2] - i[seq_length - 1] * c_[seq_length - 1]
    lambda_new = lambda9 + rho9 * temp
    return lambda_new


def update_lambda10(seq_length, h, o, c, lambda10, rho10):
    temp = h[seq_length - 1] - o[seq_length - 1] * torch.tanh(c[seq_length - 1]).to(device)
    lambda_new = lambda10 + rho10 * temp
    return lambda_new


def update_lambda11(seq_length, y, w, h, b, lambda11, rho11):
    temp = y - torch.matmul(h[seq_length - 1], w).to(device) - b
    lambda_new = lambda11 + rho11 * temp
    return lambda_new











