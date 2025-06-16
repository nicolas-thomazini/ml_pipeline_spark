import argparse
import os
import pickle
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

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

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--time-step", type=int, default=30)
parser.add_argument("--output", type=str, default="models/model_lstm_solana.bin")  # <- novo caminho
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../data/solana.csv")

TIME_STEP = args.time_step
INPUT_SHAPE = (TIME_STEP, 1)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df[["Close"]]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

x, y = create_dataset(df_scaled, TIME_STEP)
x = x.reshape(-1, TIME_STEP, 1)

with mlflow.start_run():
    model = build_model(INPUT_SHAPE)
    model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    last_seq = df_scaled[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    pred = model.predict(last_seq, verbose=0)
    pred_price = scaler.inverse_transform([[pred[0][0]]])[0][0]

    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("time_step", TIME_STEP)
    mlflow.log_metric("predicted_price_next_day", pred_price)

    print(f"📈 Preço previsto para o próximo dia: ${pred_price:.2f}")

    real = 102.30
    mape = abs(pred_price - real) / real * 100
    mlflow.log_metric("mape_vs_real", mape)
    print(f"❗ Erro percentual: {mape:.2f}%")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "wb") as f_out:
        pickle.dump((scaler, model), f_out)
    print(f"✅ Modelo salvo em: {args.output}")
    mlflow.log_artifact(args.output)

    from mlflow.models.signature import infer_signature
    signature = infer_signature(x, model.predict(x))

    mlflow.keras.log_model(
        model,
        name="model",
        registered_model_name="solana_lstm_model",
        signature=signature
    )