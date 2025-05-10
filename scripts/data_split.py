import sys
import os
# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"File directory: {os.path.dirname(__file__)}")
print(f"Project root set to: {project_root}")

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import yaml
import logging



# Load Configuration
config_path = os.path.join(project_root, 'config/data_prep_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configure MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

# Configure Logging
log_file = os.path.join(project_root, config['logging']['log_dir'], 'data_prep.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataSplitter:
    def __init__(self):
        self.input_file = os.path.join(project_root, config['data']['raw_data_path'])
        self.output_dir = os.path.join(project_root, config['data']['processed_data_path'])
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_clean_data(self):
        logging.info("Loading raw data from %s", self.input_file)
        df = pd.read_csv(self.input_file, encoding='ISO-8859-1')
        df['Sentence #'] = df['Sentence #'].ffill()
        df.drop(columns=['POS'], inplace=True)
        logging.info("Data loaded with %d sentences", len(df['Sentence #'].unique()))
        return df

    def split_data(self, df):
        test_size = config['split']['test_size']
        val_size = config['split']['val_size']
        sentences = df.groupby('Sentence #').agg({'Word': list, 'Tag': list}).reset_index()
        train, test = train_test_split(sentences, test_size=test_size, random_state=42)
        train, val = train_test_split(train, test_size=val_size, random_state=42)
        logging.info("Data split into train: %d, val: %d, test: %d", len(train), len(val), len(test))
        return train, val, test

    def save_data(self, train, val, test):
        train_file = os.path.join(self.output_dir, config['data']['train_file'])
        val_file = os.path.join(self.output_dir, config['data']['val_file'])
        test_file = os.path.join(self.output_dir, config['data']['test_file'])
        train.to_pickle(train_file)
        val.to_pickle(val_file)
        test.to_pickle(test_file)
        logging.info("Processed data saved: %s, %s, %s", train_file, val_file, test_file)

    def log_stats(self, df, train, val, test):
        mlflow.log_param("total_sentences", len(df['Sentence #'].unique()))
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("val_size", len(val))
        mlflow.log_param("test_size", len(test))
        mlflow.log_artifact(log_file)


def main():
    with mlflow.start_run():
        splitter = DataSplitter()
        df = splitter.load_and_clean_data()
        train, val, test = splitter.split_data(df)
        splitter.save_data(train, val, test)
        splitter.log_stats(df, train, val, test)
        logging.info("Data preparation completed successfully.")


if __name__ == "__main__":
    main()
