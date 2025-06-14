import pandas as pd
import json

df = pd.read_csv('./data/solana.csv')
last_30 = df['Close'].tail(30).tolist()

with open('examples/example_request.json', 'w') as f:
    json.dump({'close_values': last_30}, f, indent=2)

print('✅ example_request.json atualizado com dados reais.')
