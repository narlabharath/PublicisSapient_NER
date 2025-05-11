# Named Entity Recognition (NER) using BiLSTM-CRF

This repository contains a comprehensive implementation of **Named Entity Recognition (NER)** using the **BiLSTM-CRF** architecture. The project is designed to demonstrate a robust and scalable approach to NER, leveraging modern machine learning practices, including **MLflow for experiment tracking**, structured **logging**, and **efficient data handling**.

## 📌 Why this Repository?

The primary objective of this project is to develop a **highly efficient and scalable NER model** using a combination of **Bidirectional LSTM (BiLSTM)** and **Conditional Random Fields (CRF)**. The project also aims to demonstrate **best practices in ML engineering**, including:

* Efficient data processing and batch management.
* Robust model training with GPU acceleration.
* Automated experiment tracking using **MLflow**.
* Modular and reusable code structure.
* Comprehensive evaluation and visualization of model performance.

## 🚀 Project Structure

The project is structured into modular components for ease of maintenance and scalability:

```
├── config
│   └── bilstm_crf_config.yaml         # Configuration file for model parameters, logging, and paths
├── data
│   ├── processed                     # Preprocessed training, validation, and test data
│   └── raw                           # Raw input data
├── logs                              # Logging directory for tracking training and evaluation
├── models                            # Saved trained models
├── results                           # Results including plots and metrics
├── scripts
│   ├── bilstm_crf_data_processing.py  # Data preprocessing script
│   ├── bilstm_crf_training.py         # Training script for BiLSTM-CRF model
│   ├── evaluate_bilstm_crf.py         # Evaluation script for performance metrics
├── src
│   ├── models
│   │   └── bilstm_crf_model.py        # BiLSTM-CRF model architecture
│   └── utils
│       └── bilstm_crf_data_loader.py  # Efficient DataLoader for batch processing
└── requirements.txt                  # Dependency file
```

## 💡 How it Works

### Step 1: Data Preprocessing

* The raw data is cleaned and processed using the **bilstm\_crf\_data\_processing.py** script.
* Generates processed data stored in **data/processed/**.
* Precomputes large batches for efficient GPU usage.
* Logs data statistics and processing time using **MLflow**.

### Step 2: Model Training

* The **bilstm\_crf\_training.py** script trains the BiLSTM-CRF model.
* Uses **torch.cuda.amp** for **mixed-precision training**, significantly improving training speed.
* Incorporates **gradient clipping** to prevent exploding gradients.
* Uses **disk-based batch loading** to minimize memory usage and speed up training.
* Logs training metrics and model checkpoints using **MLflow**.

### Step 3: Evaluation and Visualization

* The **evaluate\_bilstm\_crf.py** script evaluates the model on train, validation, and test sets.
* Generates metrics including **F1 score**, **accuracy**, and **classification report**.
* Plots **F1 scores** for different data splits for visual analysis.
* Logs detailed evaluation metrics.

## ⚙️ Experiment Tracking with MLflow

All key metrics, including **loss, F1 score, and accuracy**, are logged to **MLflow**. To view the MLflow UI, use:

```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
```

MLflow is configured to save model artifacts and logs to the **mlruns/** directory. The tracking URI is dynamically set during training.

## 📝 Logging and Monitoring

* Logs are stored in the **logs/** directory.
* Detailed logs are maintained for each step: preprocessing, training, and evaluation.
* Log files include both **console outputs** and **MLflow metrics**.

## 🧩 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Preprocess the data:

```bash
python scripts/bilstm_crf_data_processing.py
```

3. Train the model:

```bash
python scripts/bilstm_crf_training.py
```

4. Evaluate the model:

```bash
python scripts/evaluate_bilstm_crf.py
```

## 💾 Model Saving and Loading

* Models are saved in the **models/** directory after training.
* Model artifacts are also logged to **MLflow** for easy versioning.

## 📊 Results and Visualization

* Results including plots and classification reports are saved in the **results/** directory.
* F1 score plots can be found as **bilstm\_crf\_f1\_scores.png**.

## 🌟 Key Features

* **Dynamic Data Loading:** Efficiently handles large datasets using **disk-based batch loading**.
* **GPU Acceleration:** Leverages **mixed precision training** to speed up computations.
* **Robust Evaluation:** Comprehensive metrics and visualizations for performance analysis.
* **Experiment Tracking:** Uses **MLflow** for reproducibility and transparency.

## 🔄 Future Improvements

* Integrating more advanced NER architectures like **Transformer-based models**.
* Adding a **UI for real-time NER prediction** using **Streamlit**.
* Extending the pipeline for **multi-language support**.

## 🤝 Contributions

Contributions are welcome! Feel free to submit issues or pull requests to enhance the project.
