# ml_solana_predict

**`ml_solana_predict`** is a machine learning pipeline for time series forecasting of the closing price of the Solana (SOL) cryptocurrency using LSTM (Long Short-Term Memory) models. It includes training, inference (via API), experiment tracking with MLflow, and utility scripts for deployment and reproducibility.

---

## 1. 🎯 Project Purpose

The goal of this project is to:

- Forecast future closing prices for the Solana cryptocurrency using historical data in notebook and other metrics.
- Provide an API endpoint to serve predictions via a Flask + Gunicorn application.
- Support model experimentation and tracking with MLflow.
- Ensure modularity, scalability, and CI/CD readiness via a clean project structure.

---

## 2. 📁 Project Structure

```bash
ml_solana_predict/
│
├── data/
│   └── solana.csv
│   └── solana.csv.dvc          # When you pull solana.csv
├── examples/                   # Sample JSON input for prediction
│   └── example_request.json    # Dataset folder (e.g., solana.csv)
├── src/
│   ├── api/                    # API (Flask app with prediction endpoint)
│   │   └── predict.py
│   ├── configs/                # Configurations (e.g., Spark or env setup)
│   ├── experiments/            # MLflow-based experiments
│   │   └── train_mlflow.py
│   ├── models/                 # Serialized models (e.g., model_lstm_solana.bin)
│   ├── pipelines/              # Utility pipeline scripts
│   │   └── pipeline_final.py
│   ├── tests/                  # Unit and integration tests
│   │   └── test_predict.py
│   └── train/                  # Traditional model training script
│       └── train.py
│       └── __init__.py         # Initialization __init__.py
├── notebooks/                  # Notebook for visualization data and other metrics
│   └── ml_pipeline.py          # Pipeline
│   └── ml_sol_notebook.ipynb   # Notebook to explore solana.csv
│   └── predict_day.csv         # Predict the next day of last close day in dataset
│
├── venv                        # If you likes use a venv
├── .dockerignore               # Docker ignore
├── .gitignore                  # Git Ignore
├── makefile                    # Automation commands (lint, format, train, etc.)
├── requirements.txt            # Python dependencies
├── README.md                   # Setup and instructions of the project
└── .flake8                     # Linting rules
```

---

## 3. 🛠️ Technologies Used

**Python 3.12**

- Flask + Gunicorn – serving the prediction API

- Keras + TensorFlow – LSTM model training

- MLflow – experiment tracking and model logging

- Scikit-learn – preprocessing (MinMaxScaler)

- Makefile – task automation (make train, make lint, etc.)

- curl / HTTP – testing the API locally

---

## 4. 🚀 How to Run the Project

To run the project you need the dataset that used. I use DVC to get the dataset, but if you didn't gets the dvc file, you can use the dataset that are on `data/solana.csv`.

1. Clone the repository:

   ```bash
   git clone https://github.com/nicolasthomazini/ml_solana_predict.git
   cd ml_solana_predict
   ```

`IF YOU WANT YOU CAN SKIP THIS PART`: This project uses **DVC (Data Version Control)** to manage and version large data files, including the solana.csv dataset stored remotely on Google Drive.

Steps to download solana.csv via DVC:

2. Install DVC:

   Make sure you have DVC installed. If not, install it with:

   ```bash
   pip install dvc[gdrive]
   ```

3. Pull the data file from remote storage:

   The dataset solana.csv is versioned and stored on Google Drive.

   If you want to access the dataset using a Google Service Account, configure DVC like this (assuming your credentials file is credentials.json at the project root):

   ```bash
   dvc remote modify gdrive_remote gdrive_use_service_account true --local
   dvc remote modify gdrive_remote gdrive_service_account_json_file_path ./credentials.json --local
   ```

   Make sure the Google Drive folder where the dataset is stored is shared with your service account email or your Google account has access.

   ```bash
   dvc remote add -d gdrive_remote:<your-hash-folder-gdrive> # you need add the dataset a folder on gdrive and share the link.
   ```

   Once configured, you can download the dataset file locally by running:

   ```bash

   dvc pull data/solana.csv
   ```

   This command fetches solana.csv from the configured Google Drive remote and saves it in the data/ folder.

4. **Verify the file exists locally**:

   After the pull completes, check that data/solana.csv is available in your project directory.

### ✅ 📦 4.1 Install Dependencies

First, create and activate a virtual environment, then install the requirements:

```bash
python -m venv venv        # Create a virtual environment (optional)
source venv/bin/activate   # Activate it (Linux/macOS)
# .\venv\Scripts\activate  # Activate it (Windows)

pip install -r requirements.txt
```

### 🧠 4.2 Train the model

`⚠️ ATTENTION`: Make sure you're at the root of the project directory before running these commands.

To train the model using the standard script:

```bash
make train
```

Or MLflow-based training (which logs metrics and artifacts):

```bash
make train-mlflow
```

To open the MLflow UI:

make mlflow-ui

`MLflow` runs a web-based user interface that allows you to track and visualize your machine learning experiments. By default, the MLflow UI server runs locally on port 5000, accessible via http://localhost:5000. This makes it easy to monitor training runs and compare models through your browser.

You can start the MLflow UI with the provided make command, and if needed, customize the port or host by passing additional parameters. This flexibility allows you to run multiple MLflow servers or integrate MLflow into different environments seamlessly.

---

## 🔮 4.3 Run the API

To start the API locally with Gunicorn:

```bash
make run-api
```

---

### 🔍 4.4 Run tests & lint

You can test the project.

`make lint` it's a command that find error or bad formatations if exists in the files. You can run it and next run the `make format`, that format the code.

```bash
make test      # Run unit tests
make lint      # Check style using flake8
make format    # Auto-format code with black + isort
```

---

## 5. 🐳 #TODO: Run with Docker

A docker-compose.yml file is being prepared to launch both:

- the prediction API (Flask + Gunicorn)

- the MLflow tracking server

You will soon be able to run everything with:

```
docker-compose up --build
```

✅ This will ensure full reproducibility and easier deployment.

## License

This project is licensed under the [MIT License](License-MIT). See LICENSE for details.

## 7. 🧠 Contributing & Ideas

This project is a learning-friendly and scalable ML pipeline. If you want to:

- Add more advanced models (e.g., GRU, Prophet, etc.)

- Improve data preprocessing

- Add real-time data fetching

- Expand to multi-asset forecasting

- Feel free to fork and open a PR or suggestion.
