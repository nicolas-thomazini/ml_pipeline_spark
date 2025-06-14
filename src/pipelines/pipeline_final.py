import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Input  # type: ignore
from keras.models import Sequential  # type: ignore
from pyspark.sql import SparkSession
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K  # type: ignore

spark = SparkSession.builder.appName("Solana ML Pipeline").getOrCreate()

df_spark = spark.read.csv(
    "file:///home/nicolas/ml_pipeline_spark/data/solana.csv",
    header=True,
    inferSchema=True,
)

df_real = df_spark.toPandas()
df_real["Date"] = pd.to_datetime(df_real["Date"], format="%Y-%m-%d")
df_real.set_index("Date", inplace=True)
df_real = df_real[["Close"]]

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_real)


def create_dataset(data, time_step=1):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i : (i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)


time_step = 30
x, y = create_dataset(df_scaled, time_step)
x = x.reshape(x.shape[0], x.shape[1], 1)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

K.clear_session()
model = Sequential()
model.add(Input(shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")
history = model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

y_pred = model.predict(x_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled) * 100
print(f"📊 MAPE: {mape:.2f}% between predicted and real closing prices")

last_data = df_scaled[-time_step:].reshape(1, time_step, 1)
predicted_next_day = model.predict(last_data)
predicted_next_day_rescaled = scaler.inverse_transform(predicted_next_day)

predicted_day_df = pd.DataFrame(
    {
        "Date": [df_real.index[-1] + pd.Timedelta(days=1)],
        "Predicted_Close": predicted_next_day_rescaled.flatten(),
    }
)
predicted_day_df.to_csv("predict_day.csv", index=False)

df_forecast = pd.DataFrame(
    {
        "Date": df_real.index[-len(y_test):],
        "Real": y_test_rescaled.flatten(),
        "Predicted": y_pred_rescaled.flatten(),
    }
)
df_forecast.to_csv("forecast_solana.csv", index=False)

df_real["MA7"] = df_real["Close"].rolling(window=7).mean()
df_real["MA20"] = df_real["Close"].rolling(window=20).mean()

plt.figure(figsize=(12, 6))
plt.plot(df_real.index, df_real["Close"], label="Real Closing", color="gray")
plt.plot(
    df_forecast["Date"], df_forecast["Predicted"], label="Predict (LSTM)", color="red"
)
plt.plot(df_real.index, df_real["MA7"], label="MA 7 days", color="yellow")
plt.plot(df_real.index, df_real["MA20"], label="MA 20 days", color="orange")

plt.title("Closing Price Prediction for Solana with LSTM")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
