import sys
import os
# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"File directory: {os.path.dirname(__file__)}")
print(f"Project root set to: {project_root}")


import joblib
import mlflow
import yaml
import logging
import pandas as pd
from src.models.crf_model import CRFModel


# Load Configuration
config_path = os.path.join(project_root, 'config/crf_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configure MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

# Configure Logging
log_file = os.path.join(project_root, config['logging']['log_dir'], 'train_crf.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
# logging.basicConfig(
#     filename=log_file,
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

logger = logging.getLogger("train_crf")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("Logger initialized, log file should now exist.")
for handler in logger.handlers:
    handler.flush()

print(f"Log file created at: {log_file}")


# Load processed data
processed_data_path = os.path.join(project_root, config['data']['processed_data_path'])
train_data_path = os.path.join(processed_data_path, 'crf_train.pkl')
(X_train, y_train) = joblib.load(train_data_path)

# Initialize CRF Model
model = CRFModel()

def train_crf():
    with mlflow.start_run():
        try:
            logger.info("Starting CRF model training...")
            mlflow.log_param("model_type", "CRF")
            mlflow.log_param("train_data_size", len(X_train))
            model.train(X_train, y_train)
            model_save_path = os.path.join(project_root, config['deployment']['model_save_path'], 'crf_model.joblib')
            model.save_model(model_save_path)  
            mlflow.log_artifact(log_file)
            logger.info("CRF model training completed successfully.")
        except Exception as e:
            logger.error("Error during CRF model training: %s", str(e))
            mlflow.log_param("training_status", "failed")

if __name__ == "__main__":
    train_crf()
