# ml_solana_predict

ml_solana_predict is a machine learning pipeline for forecasting the closing price of the Solana (SOL) cryptocurrency using LSTM (Long Short-Term Memory) models. It includes:

Training scripts (traditional and with MLflow)

An API for model inference

MLflow for experiment tracking

Utility scripts for reproducibility and automation

## 1. 🎯 Project Purpose

This project was built to:

- Forecasting SOL’s next-day closing price using LSTM

- Model deployment via an API endpoint

- Experiment tracking and reproducibility with MLflow

- Execution either locally (Makefile) or via Docker

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
├── models/
│   └── model_lstm_solana.bin   # Serialized models (e.g., model_lstm_solana.bin)
├── src/
│   ├── api/                    # API (Flask app with prediction endpoint)
│   │   └── predict.py
│   ├── configs/                # Configurations (e.g., Spark or env setup)
│   ├── experiments/            # MLflow-based experiments
│   │   └── train_mlflow.py
│           ├── models/                
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

- Keras + TensorFlow (LSTM model)

- Flask + Gunicorn (API)

- MLflow (tracking + serving)

- Scikit-learn (scaling)

- DVC (optional, for data versioning)

- Docker + Docker Compose

- Makefile (local task automation)

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

---

### 4.2

You can choose between:

### 🧠 4.2.1 Train the model

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

## 🔮 4.2.2 Run the API

To start the API locally with Gunicorn:

```bash
make run-api
```

---

### 🔍 4.2.3 Run tests & lint

You can test the project.

`make lint` it's a command that find error or bad formatations if exists in the files. You can run it and next run the `make format`, that format the code.

```bash
make test      # Run unit tests
make lint      # Check style using flake8
make format    # Auto-format code with black + isort
```

---

## 4.3 🐳 Dockerized (MLflow + Serving Ready)

Run everything — MLflow UI + Serving API — via Docker Compose:

```bash
docker-compose up --build
```

This will:

- 🧪 Start the MLflow Tracking UI at http://localhost:5000

- 🤖 Start the MLflow Model Serving API at http://localhost:5001/invocations

- 🗂️ Mount volumes so models and logs persist in ./mlruns/

Ensure your model is registered in MLflow as solana_lstm_model (done automatically via train_mlflow.py).

## 6. 🔧 Dev Tools

```bash
make lint      # Check style (flake8)
make format    # Auto-format code
make clean     # Remove temporary files
```

✅ This will ensure full reproducibility and easier deployment.

## 7. 🧠 Contributing & Ideas

This project is a learning-friendly and scalable ML pipeline. If you want to:

- Add more advanced models (e.g., GRU, Prophet, etc.)

- Improve data preprocessing

- Add real-time data fetching

- Expand to multi-asset forecasting

- Feel free to fork and open a PR or suggestion.

## License

This project is licensed under the [MIT License](License-MIT). See LICENSE for details.

