#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import pickle

from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

TIME_STEP = 30
EPOCHS = 20
BATCH_SIZE = 32
INPUT_SHAPE = (TIME_STEP, 1)
DATA_PATH = "./data/solana.csv"
OUTPUT_MODEL_PATH = "./src/models/model_lstm_solana.bin"


def create_dataset(data, time_step=30):
    x, y = [], []
    for i in range(len(data) - time_step):
        x.append(data[i : i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)


def build_model(input_shape):
    K.clear_session()
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


print("🔄 Carregando e preparando os dados...")
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df[["Close"]]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

x, y = create_dataset(df_scaled, TIME_STEP)
x = x.reshape(-1, TIME_STEP, 1)

print(f"🧠 Treinando modelo LSTM em {len(x)} amostras...")
model = build_model(INPUT_SHAPE)
model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

last_sequence = df_scaled[-TIME_STEP:].reshape(1, TIME_STEP, 1)
predicted_next = model.predict(last_sequence, verbose=0)
predicted_price = scaler.inverse_transform([[predicted_next[0][0]]])[0][0]

print(f"📈 Preço previsto para o próximo dia: ${predicted_price:.2f}")

real_value_next_day = 102.30
error = abs(predicted_price - real_value_next_day) / real_value_next_day
print(
    f"❗ Erro percentual em relação ao valor real (${real_value_next_day}): {error * 100:.2f}%"
)

with open(OUTPUT_MODEL_PATH, "wb") as f_out:
    pickle.dump((scaler, model), f_out)

print(f"✅ Modelo e scaler salvos em: {OUTPUT_MODEL_PATH}")
