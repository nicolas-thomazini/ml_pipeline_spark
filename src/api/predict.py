import pickle
import numpy as np
import os 

import pandas as pd
from flask import Flask, request, jsonify

# Parâmetros
time_step = 30

current_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(current_dir, '..', '..', 'src/models/model_lstm_solana.bin')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Carregar scaler e modelo
with open(model_file, 'rb') as f_in:
    scaler, model = pickle.load(f_in)

# Inicializando o Flask
app = Flask('solana_predict')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Verifica se o corpo da requisição tem os dados corretos
    if not data or 'close_values' not in data:
        return jsonify({'error': 'JSON deve conter a chave "close_values" com uma lista de 30 valores.'}), 400

    close_values = data['close_values']

    if len(close_values) != time_step:
        return jsonify({'error': f'É necessário fornecer exatamente {time_step} valores de fechamento.'}), 400

    # Converte para DataFrame e normaliza
    df_input = pd.DataFrame(close_values, columns=['Close'])
    df_scaled = scaler.transform(df_input)

    # Prepara os dados para o modelo (reshape 3D)
    input_data = np.array(df_scaled).reshape(1, time_step, 1)

    # Faz a predição
    y_pred = model.predict(input_data)

    # Desfaz a normalização para obter o valor de fechamento previsto
    y_pred_rescaled = scaler.inverse_transform([[y_pred[0][0]]])

    result = {
        'predicted_close': float(y_pred_rescaled[0][0])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
