import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

TIME_STEP = 30
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "model_lstm_solana.bin"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with open(MODEL_PATH, "rb") as f:
    scaler, model = pickle.load(f)

app = Flask("solana_predict")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "close_values" not in data:
        return (
            jsonify(
                {
                    "error": 'JSON deve conter a chave "close_values" com uma lista de 30 valores.'
                }
            ),
            400,
        )

    close_values = data["close_values"]

    if len(close_values) != TIME_STEP:
        return (
            jsonify(
                {
                    "error": f"É necessário fornecer exatamente {TIME_STEP} valores de fechamento."
                }
            ),
            400,
        )

    try:
        df_input = pd.DataFrame(close_values, columns=["Close"])
        df_scaled = scaler.transform(df_input)
        input_data = np.array(df_scaled).reshape(1, TIME_STEP, 1)

        y_pred = model.predict(input_data)
        y_pred_rescaled = scaler.inverse_transform([[y_pred[0][0]]])

        return jsonify({"predicted_close": float(y_pred_rescaled[0][0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
