# -*- coding: utf-8 -*-
# @Time    : 2023/1/15 16:19
# @Author  : liushuo
# @FileName: admm_lstm.py
# @Software: PyCharm

"""
  admm_lstm.py (Original: main.py by Liu, et al.)
  Original author: Liu Shuo, et al.
  Modified by: (c) Yang Yang, 2024-2025
"""

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义超参数
epoch = 200
# batch_size = 60000
time_step = 10
input_size = 28

# 定义网络参数
num_classes = 1
hidden_size = 10
num_layers = 1


# def cross_entropy_with_softmax(label, a):
#     prob = softmax(a)
#     imask = torch.eq(prob, 0.0).to(device)
#     prob = torch.where(imask, torch.tensor(1e-10, device=device).to(device), prob)
#     loss = cross_entropy(label, prob)
#     return loss


# def softmax(x):
#     exp =torch.exp(x).to(device)
#     imask =torch.eq(exp,float("inf")).to(device)
#     exp = torch.where(imask,torch.exp(torch.tensor(88.6,device=device)),exp).to(device)
#     exp = exp.transpose(0,1)
#     # print(exp.size())
#     temp = torch.sum(exp,dim=0).to(device)+1e-10
#     temp2 = exp/temp
#     temp2 = temp2.transpose(0,1)
#     return temp2


# def cross_entropy(label, prob):
#     loss = -torch.sum(label * torch.log(prob)).to(device)
#     return loss


def sigmoid(x):
    return 1. / (1 + torch.exp(-x).to(device))


def sigmoid_der(x):
    with torch.no_grad():
        values = torch.sigmoid(x)
    return values*(1-values)


def tanh_der(x):
    with torch.no_grad():
        values = torch.tanh(x).to(device)
    return 1. - values ** 2


def fro_qua(x):
    with torch.no_grad():
        return torch.sum(x * x)


# def update_Wy(a, Wy, h, lambda1, lambda11, rho11):
#     with torch.no_grad():
#         Form10 = rho11 * (torch.matmul(torch.t(h), a + lambda11 / rho11))
#         Form11 = lambda1 * torch.eye(h.size(1)) + rho11 * torch.matmul(torch.t(h), h)
#         Wy_new = torch.matmul(torch.inverse(Form11), Form10)
#     return Wy_new
def eq1_W(a, Wy, h, lambda11, rho11):
    """ 计算Wy的梯度 """
    temp = a - torch.matmul(h, Wy).to(device) + lambda11/rho11
    temp2 = torch.transpose(h,0,1)
    res = rho11 * torch.matmul(temp2, temp)
    return res

def eq1(a, Wy, h, lambda11, rho11):
    temp = a - torch.matmul(h, Wy) + lambda11/rho11
    res = rho11/2 * torch.sum(temp * temp)
    return res

def P(W_new, theta, a, W, h, lambda11, rho11):
    temp = W_new - W
    res = eq1(a, W, h, lambda11, rho11) - torch.sum(eq1_W(a, W, h, lambda11, rho11) *temp) + torch.sum(theta * temp * temp) / 2
    return res

def update_Wy(a, Wy, h, lambda1, lambda11, rho11):
    with torch.no_grad():
        gradients = eq1_W(a, Wy, h, lambda11, rho11)
        theta_Wy = 0.01
        zeta = Wy + gradients / theta_Wy
        while (eq1(a, zeta, h, lambda11, rho11) > P(zeta, theta_Wy, a, Wy, h, lambda11, rho11)):
            theta_Wy = theta_Wy * 2
            zeta = Wy + gradients / theta_Wy
        Wy_new = zeta
    return Wy_new


def update_W(z, W, x, U, h, lambda0, LAMBDA, RHO, T):
    with torch.no_grad():
        Func1, Func2 = 0, 1
        theta_W = 1
        while Func2 > Func1:
            Form11, Form12, Form21 = 0, 0, 0
            for t in range(T):
                Form10 = -z[t] + torch.matmul(x[:,t,:], W).to(device) + torch.matmul(h[t-1], U).to(device) - LAMBDA[:,t,:]/RHO
                Form11 = Form11 + RHO * torch.sum(Form10 * Form10).to(device)
                Form12 = Form12 + RHO * torch.matmul(torch.t(x[:,t,:]).to(device), Form10).to(device)
            # W1 = (theta_W * W + RHO * Form12) / (lambda0 + theta_W)
            W1 = W - Form12 / theta_W  # From12是梯度（在W处）
            Form12 = torch.sum(Form12 * (W1 - W))  # 梯度和差值的内积
            Form13 = torch.sum((W1 - W)*(W1 - W))  # 差值的平方项
            Func1 = 0.5 * Form11 + Form12 + 0.5 * theta_W * Form13
            for t in range(T):
                Form20 = -z[t] + torch.matmul(x[:,t,:], W1).to(device) + torch.matmul(h[t-1], U) - LAMBDA[:,t,:] / RHO
                Form21 = Form21 + RHO * torch.sum(Form20 * Form20).to(device)
            Func2 = 0.5 * Form21
            theta_W = theta_W * 2
        theta_W = theta_W/2
        Temp12 = 0
        for t in range(T):
            Temp10 = -z[t] + torch.matmul(x[:, t, :], W).to(device) + torch.matmul(h[t - 1], U).to(device) - LAMBDA[:, t, :] / RHO
            Temp12 = Temp12 + RHO * torch.matmul(torch.t(x[:, t, :]).to(device), Temp10).to(device)
        W_new = (theta_W * W - Temp12) / (lambda0 + theta_W)
    return W_new


def update_U(z, W, x, U, h, lambda0, LAMBDA, RHO, T):
    with torch.no_grad():
        Func1, Func2 = 0, 1
        theta_U = 1
        while Func2 > Func1:
            Form11, Form12, Form21 = 0, 0, 0
            for t in range(T):
                Form10 = -z[t] + torch.matmul(x[:,t,:], W).to(device) + torch.matmul(h[t - 1], U).to(device) - LAMBDA[:,t,:] / RHO
                Form11 = Form11 + RHO * torch.sum(Form10 * Form10).to(device)
                Form12 = Form12 + RHO * torch.matmul(torch.t(h[t-1]).to(device), Form10).to(device)
            U1 = U -  Form12 / theta_U
            Form12 = torch.sum(Form12 * (U1 - U))
            Form13 = torch.sum((U1 - U) * (U1 - U))
            Func1 = 0.5 * Form11 + Form12 + 0.5 * theta_U * Form13
            for t in range(T):
                Form20 = z[t] - torch.matmul(x[:,t,:], W).to(device) - torch.matmul(h[t - 1], U1) + LAMBDA[:,t,:] / RHO
                Form21 = Form21 + RHO *torch.sum(Form20 * Form20).to(device)
            Func2 = 0.5 * Form21
            theta_U = theta_U * 2
        theta_U = theta_U/2
        Temp12 = 0
        for t in range(T):
            Temp10 = -z[t] + torch.matmul(x[:, t, :], W).to(device) + torch.matmul(h[t - 1], U).to(device) - LAMBDA[:, t, :] / RHO
            Temp12 = Temp12 + RHO *torch.matmul(torch.t(h[t - 1]).to(device), Temp10).to(device)
        U_new = (theta_U * U - Temp12) / (lambda0 + theta_U)
    return U_new


def update_z(z, output, W, x, U, h, lambda1, rho1, lambda2, rho2):
    with torch.no_grad():
        temp = torch.max(torch.abs(output - lambda2 / rho2)).to(device)
        appro_z = 0.5 * (1 + temp) + 0.125
        Form1 = torch.matmul(x, W).to(device) + torch.matmul(h, U).to(device) - lambda1 / rho1
        Form2 = rho2 * (torch.sigmoid(z) - output + lambda2 / rho2) * sigmoid_der(z)
        Form3 = rho1 * Form1 + 0.5 * rho2 * appro_z * z - Form2
        iterz = 2 * Form3 / (2 * rho1 + rho2 * appro_z)
        return iterz


def update_zg(z, output, W, x, U, h, lambda1, rho1, lambda2, rho2):
    with torch.no_grad():
        temp = torch.max(torch.abs(output - lambda2 / rho2)).to(device)
        appro_zg = 2 * (1 + temp) + 2
        Form1 = torch.matmul(x, W).to(device) + torch.matmul(h, U).to(device) - lambda1 / rho1
        Form2 = rho2 * (torch.tanh(z) - output + lambda2 / rho2) * tanh_der(z)
        Form3 = rho1 * Form1 + 0.5 * rho2 * appro_zg * z - Form2
        iterzg = 2 * Form3 / (2 * rho1 + rho2 * appro_zg)
    return iterzg


def update_f(zf, g, i, ct, ct_, lambda2, rho2, lambda9, rho9):
    with torch.no_grad():
        Form1 = rho2 * (torch.sigmoid(zf) + lambda2 / rho2) + rho9 * ct_ * (ct - g * i + lambda9 / rho9)
        Form2 = rho2 + rho9 * ct_ * ct_
        iterf = Form1 / Form2
    return iterf


def update_i(zi, g, f, ct, ct_, lambda4, rho4, lambda9, rho9):
    with torch.no_grad():
        Form1 = rho4 * (torch.sigmoid(zi) + lambda4 / rho4) + rho9 * g * (ct - ct_ * f + lambda9 / rho9)
        Form2 = rho4 + rho9 * g * g
        iteri = Form1 / Form2
    return iteri


def update_o(zo, ct, h, lambda6, rho6, lambda10, rho10):
    with torch.no_grad():
        Form1 = rho6 * (torch.sigmoid(zo) + lambda6 / rho6) + rho10 * torch.tanh(ct) * (h - lambda10 / rho10)
        Form2 = rho6 + rho10 * torch.tanh(ct) * torch.tanh(ct)
        itero = Form1 / Form2
    return itero


def update_g(zg, i, f, ct, ct_, lambda8, rho8, lambda9, rho9):
    with torch.no_grad():
        Form1 = rho8 * (torch.tanh(zg) + lambda8 / rho8) + rho9 * i * (ct - ct_ * f + lambda9 / rho9)
        Form2 = rho8 + rho9 * i * i
        iterg = Form1 / Form2
    return iterg


def update_c(f, i, o, g, ct, ct_, h, lambda9, rho9, lambda10, rho10):
    with torch.no_grad():
        temp = torch.max(torch.abs((h - lambda10 / rho10) * 1 / o)).to(device)
        appro_h = 2 * (1 + temp)+2
        Form1 = rho9 * (g * i + ct_ * f - lambda9 / rho9)
        Form2 = rho10 * (torch.tanh(ct) * o - h + lambda10 / rho10) * tanh_der(ct) * o

        # Form3 = 0.5 * rho10 * o * o * ct * appro_h
        # Form4 = rho9 + 0.5 * rho10 * o * o * appro_h

        qua_o = fro_qua(o)
        Form3 = 0.5 * rho10 * qua_o * ct * appro_h
        Form4 = rho9 + 0.5 * rho10 * qua_o * appro_h

        iterc = (Form1 - Form2 + Form3) / Form4
    return iterc


def update_h(c, o, a, Wy, h, lambda10, rho10, lambda11, rho11, t, T):
    with torch.no_grad():
        theta_h = 1
        Form1 = rho10 * (torch.tanh(c) * o + lambda10 / rho10)
        Func1, Func2 = 0, 1
        if t < T - 1:
            h_new = Form1 / rho10
        else:
            while Func2 > Func1:
                Form10 = -a + torch.matmul(h, Wy).to(device) - lambda11 / rho11
                Form11 = torch.matmul(Form10, torch.t(Wy).to(device)).to(device)
                h1 = h - rho11 * Form11/theta_h
                Form12 = (h1 - h) * (h1 - h)
                Func1 = 0.5 * rho11 * torch.sum(Form10 * Form10) + rho11 * torch.sum(Form11 * (h1 - h)) \
                        + 0.5 * theta_h * torch.sum(Form12)
                Form20 = a - torch.matmul(h1, Wy).to(device) + lambda11 / rho11
                Func2 = 0.5 * rho11 * torch.sum(Form20 * Form20)
                theta_h = theta_h * 2
            theta_h = theta_h/2
            h_new = (Form1 - rho11 * Form11 + theta_h * h) / (rho10 + theta_h)
    return h_new


def update_a(a_old, labels, Wy, h, lambda11, rho11):
    with torch.no_grad():
        temp1 = 2 * labels / 4224 + rho11 * torch.matmul(h, Wy).to(device) - lambda11
        y_old = temp1 / (2 / 4224 + rho11)  # ..
        a = y_old
    return a


def update_lambda11(a, Wy, h, lambda11, rho11):
    with torch.no_grad():
        Form1 = a - torch.matmul(h, Wy)
        itera11 = lambda11 + rho11 * Form1
    return itera11


def update_lambda10(c, o, h, lambda10, rho10, t, T):
    with torch.no_grad():
        # if t==T-1:
        #     rho10 = 0.01
        # else:
        #     rho10 = rho10
        Form1 = torch.tanh(c) * o - h
        itera10 = lambda10 + rho10 * Form1
    return itera10


def update_lambda9(ct, ct_, g, i, f, lambda9, rho9):
    with torch.no_grad():
        Form1 = ct - g*i - ct_*f
        itera9 = lambda9 + rho9 * Form1
    return itera9


def update_lambda8(zg, g, lambda8, rho8):
    with torch.no_grad():
        Form1 = torch.tanh(zg) - g
        itera8 = lambda8 + rho8 * Form1
    return itera8


def update_lambda_singular(z, W, x, U, h, lambda_singular, rho_singular):
    with torch.no_grad():
        Form1 = z - torch.matmul(x, W) - torch.matmul(h, U)
        itera_singular = lambda_singular + rho_singular * Form1
    return itera_singular


def update_lambda_plural(z, output, lambda_plural, rho_plural):
    with torch.no_grad():
        Form1 = torch.sigmoid(z) - output
        itera_plural = lambda_plural + rho_plural * Form1
    return itera_plural
