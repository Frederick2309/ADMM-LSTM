# -*- coding: utf-8 -*-
# @Time    : 2022/4/15 20:33
# @Author  : liushuo
# @FileName: LSTM_.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from numpy import concatenate
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
from Load_data import pre_data
from keras import optimizers


x_train, y_train, x_val, y_val, x_test, y_test, L_max = pre_data()
X_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
X_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
X_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
model = Sequential()
model.add(LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(y_train.shape[1]))

sgd = optimizers.SGD(lr=0.08)
model.compile(loss='mean_squared_error', optimizer=sgd)

history = model.fit(X_train, y_train, epochs=3, batch_size=24360, validation_data=0)


y_predict = model.predict(X_test)



plt.plot(history.history['loss'][1:-1], label='train')
#plt.plot(history.history['val_loss'], label='test')
# plt.title('LSTM_600000.SH', fontsize='12')
plt.ylabel('loss', fontsize='10')
plt.xlabel('epoch', fontsize='10')
plt.legend()

# plt.plot(y_test * L_max,color='red',label='Original')
# plt.plot(y_predict * L_max,color='green',label='Predict')
# plt.xlabel('the number of test data')
# plt.ylabel('earn_rate')
# plt.title('2016.3—2017.12')
# plt.legend()
plt.show()
# model = Sequential()
# model.add(LSTM(1, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
# model.add(Dense(y.shape[1]))
# model.compile(loss='mean_squared_error', optimizer='adam')
# # fit network
# for i in range(10):
#     model.fit(X, y, epochs=1, batch_size=1, verbose=0, shuffle=False)
#     model.reset_states()
# # make one forecast with an LSTM,
# def forecast_lstm(model, X, n_batch):
#     # reshape input pattern to [samples, timesteps, features]
#     X = X.reshape(1, 1, len(X))
#     # make forecast
#     forecast = model.predict(X, batch_size=n_batch)
#     # convert to array
#     return [x for x in forecast[0, :]]
#
# def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
#     forecasts = list()
#     for i in range(len(test)):
#         X, y = test[i, 0:n_lag], test[i, n_lag:]
#         # make forecast
#         forecast = forecast_lstm(model, X, n_batch)
#         # store the forecast
#         forecasts.append(forecast)
#     return forecasts
# forecasts = make_forecasts(model, 1, train, val, 1, 3)
