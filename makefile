.PHONY: help train train-mlflow run-api mlflow-ui lint test clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make train           - Treina o modelo padrão (src/models/train.py)"
	@echo "  make train-mlflow    - Treina o modelo com MLflow tracking (src/experiments/train_mlflow.py)"
	@echo "  make run-api         - Inicia a API com Gunicorn"
	@echo "  make mlflow-ui       - Inicia a interface do MLflow em http://localhost:5000"
	@echo "  make lint            - Checa estilo com flake8"
	@echo "  make format		  - Formata erros e inconsistências dos arquivos"
	@echo "  make test            - Testa a API local usando exemplo real"
	@echo "  make clean           - Remove arquivos temporários e de cache"

train:
	python src/models/train.py

train-mlflow:
	python src/experiments/train_mlflow.py --epochs 20 --batch-size 32

run-api:
	gunicorn -w 4 -b 127.0.0.1:9696 src.api.predict:app

mlflow-ui:
	mlflow ui

lint:
	flake8 src

format:
	black src
	isort src
	autoflake --remove-all-unused-imports --in-place -r src

test:
	curl -X POST http://127.0.0.1:9696/predict \
		-H "Content-Type: application/json" \
		-d @examples/example_request.json

# clear temp files
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
