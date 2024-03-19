import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import pydot_ng as pydot
import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Input, Dropout
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import mean_squared_error, mean_absolute_error
from web_scraper import prepare_data

lb = LabelBinarizer()
tf.random.set_seed(100045)
epoch_list = [
    10,
    100,
    1000,
]
linestyles = [(0, (1, 1)), (0, (5, 10)), "solid"]
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i : i + look_back, 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def create_lstm_model(input_shape):
    model = keras.Sequential()
    model.add(Input(shape=(input_shape)))
    model.add(LSTM(200, activation=tf.nn.leaky_relu))
    model.add(Dense(200, activation=tf.nn.leaky_relu))
    model.add(Dense(100, activation=tf.nn.leaky_relu))
    model.add(Dense(50, activation=tf.nn.leaky_relu))
    model.add(Dense(5, activation=tf.nn.leaky_relu))
    model.add(Dense(1, activation=tf.nn.leaky_relu))
    return model


def custom_learning_rate(epoch):
    if epoch <= 150:
        learning_rate = (10**-5) * (epoch / 150)
    elif epoch <= 400:
        initial_lrate = 10**-5
        k = 0.01
        learning_rate = initial_lrate * math.exp(-k * (epoch - 150))
    else:
        learning_rate = 10**-6
    return learning_rate


data_frame = prepare_data("TSLA")
dataset = data_frame.iloc[:, 5:6].values.astype("float32")
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# separate into train & test
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_data, test_data = dataset[test_size : len(dataset), :], dataset[0:test_size, :]

trainX, trainY = create_dataset(train_data)
testX, testY = create_dataset(test_data)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = create_lstm_model(input_shape=(trainX.shape[1], 1))
plot_model(model, show_shapes=True)
model.summary()

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])


def train_model(e_val):
    callback = LearningRateScheduler(custom_learning_rate)
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=e_val,
        batch_size=5,
        verbose=2,
        callbacks=[
            callback,
        ],
    )
    return model


fig, ax1 = plt.subplots(1, 1)
fig.set_figheight(5)
fig.set_figwidth(15)

plt.figure(figsize=(20, 7))
plt.plot(
    data_frame["date"].values[:],
    scaler.inverse_transform(dataset),
    color="black",
    label="Actual",
)

rmse_list, mape_list = [], []
testY_reshaped = scaler.inverse_transform([testY])

for e_val in epoch_list:
    # train the model
    model = train_model(e_val=int(e_val))

    historical_data = model.history.history
    loss = historical_data["loss"]
    val_loss = historical_data["val_loss"]

    # predict
    prediction = model.predict(testX)
    plt.plot(
        data_frame["date"].values[: prediction.shape[0]],
        scaler.inverse_transform(prediction),
        color="red",
        linestyle=linestyles[epoch_list.index(e_val)],
        label=f"Prediction_{e_val}",
    )

    # plot loss rate
    x = [i for i in range(1, e_val + 1, 1)]
    ax1.plot(
        x,
        loss,
        linestyle=linestyles[epoch_list.index(e_val)],
        color="black",
        label=f"Training Loss ({e_val})",
    )
    ax1.plot(
        x,
        val_loss,
        linestyle=linestyles[epoch_list.index(e_val)],
        color="black",
        label=f"Validation Loss ({e_val})",
    )

    # evaluate the performance
    testY_predict = scaler.inverse_transform(model.predict(testX))
    rmse = math.sqrt(mean_squared_error(testY_reshaped[0], testY_predict[:, 0]))
    rmse_list.append(rmse)
    mape = np.mean(
        np.abs(testY_predict[:, 0] - testY_reshaped[0]) / np.abs(testY_reshaped[0])
    )
    mape_list.append(mape)


ax1.set(xlabel="Epochs", ylabel="Loss")
ax1.legend(loc="best")
plt.xticks(
    np.arange(
        data_frame["date"][0], data_frame["date"][:].shape[0] + prediction.shape[0], 200
    )
)
plt.title("Stock Price Prediction Using LSTM")
plt.xlabel("Time")
plt.ylabel("Adj. Close Price (USD)")
# plt.legend(loc="best")

for i in range(0, len(rmse_list)):
    print(f"RMSE: {rmse_list[i]}, MAPE: {mape_list[i]}")


plt.show()
