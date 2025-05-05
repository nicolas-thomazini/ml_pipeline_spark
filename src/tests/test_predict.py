import requests

url = 'http://127.0.0.1:9696/predict'
payload = {
    'close_values': [
        200, 210, 215, 220, 210, 208, 205, 202, 198, 199,
        205, 208, 212, 215, 220, 225, 230, 234, 235, 240,
        245, 250, 255, 260, 265, 270, 275, 280, 285, 290
    ]
}

response = requests.post(url, json=payload)

assert response.status_code == 200, f"Erro na requisição: {response.text}"
result = response.json()

print("✅ Previsão retornada:", result)

assert 'predicted_close' in result, "Resposta inválida: chave 'predicted_close' ausente"
assert isinstance(result['predicted_close'], float), "Tipo inválido para previsão"
