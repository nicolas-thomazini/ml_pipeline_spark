#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import pickle
import pandas as pd
import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Input # type: ignore
from tensorflow.keras import backend as K # type: ignore

print("Dispositivos disponíveis:", tf.config.list_physical_devices())

# Hiperparâmetros
TIME_STEP = 30
N_SPLITS = 5
EPOCHS = 20
BATCH_SIZE = 32
INPUT_SHAPE = (TIME_STEP, 1)
OUTPUT_FILE = '/src/models/model_lstm_solana.bin'

# Carregando os dados
df = pd.read_csv('./data/solana.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Close']]

# Normalização
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Criação de janelas de séries temporais
def create_dataset(data, time_step=1):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

x, y = create_dataset(df_scaled, TIME_STEP)
x = x.reshape(x.shape[0], TIME_STEP, 1)

# Validação com TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
scores = []

# Função de construção do modelo
def build_model():
    K.clear_session()
    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

print(f'Doing validation with LSTM over {N_SPLITS} splits...')

# Inicializa o modelo uma única vez para evitar retracing
model = build_model()
model.predict(np.zeros((1, *INPUT_SHAPE)))  # warm-up

for i, (train_idx, val_idx) in enumerate(tscv.split(x)):
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    y_pred = model.predict(x_val, batch_size=BATCH_SIZE, verbose=0)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    scores.append(mape)

    print(f'MAPE on fold {i}: {mape:.4f}')

print(f'✅ Final MAPE: {np.mean(scores):.4f} ± {np.std(scores):.4f}')

# Treinamento final no dataset completo
print('Training final model on full dataset...')
model = build_model()
model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

with open(OUTPUT_FILE, 'wb') as f_out:
    pickle.dump((scaler, model), f_out)

print(f'✅ Model and scaler saved to {OUTPUT_FILE}')
