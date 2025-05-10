import sys
import os
import pandas as pd
import joblib
import yaml
import logging
import mlflow
from torch.nn.utils.rnn import pad_sequence
import torch

# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"File directory: {os.path.dirname(__file__)}")
print(f"Project root set to: {project_root}")


# Load Configuration
config_path = os.path.join(project_root, 'config/bilstm_crf_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

class BilstmCRFDataProcessor:
    def __init__(self):
        # Configure Logging
        log_file = os.path.join(project_root, config['logging']['log_dir'], 'bilstm_crf_data_processing.log')
        self.logger = logging.getLogger("bilstm_crf_data_processing")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Load vocab and tag mappings
        processed_data_path = os.path.join(project_root, config['training']['processed_data_path'])
        vocab_path = os.path.join(processed_data_path, 'vocab.pkl')
        tag_map_path = os.path.join(processed_data_path, 'tag_map.pkl')

        with open(vocab_path, 'rb') as f:
            self.word2idx, self.idx2word = joblib.load(f)
        with open(tag_map_path, 'rb') as f:
            self.tag2idx, self.idx2tag = joblib.load(f)

        self.logger.info("Loaded vocabulary and tag mappings.")

    def pad_sentences(self, sentences, padding_value=0):
        padded = pad_sequence([torch.tensor(sent, dtype=torch.long) for sent in sentences], batch_first=True, padding_value=padding_value)
        return padded

    def preprocess_data(self, data):
        X = []
        y = []
        for sentence, tags in zip(data['Word'], data['Tag']):
            word_indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence]
            tag_indices = [self.tag2idx.get(tag, self.tag2idx['O']) for tag in tags]
            X.append(word_indices)
            y.append(tag_indices)
        X = self.pad_sentences(X)
        y = self.pad_sentences(y)
        return X, y

    def save_processed_data(self, X, y, file_name):
        processed_data_path = os.path.join(project_root, config['training']['processed_data_path'])
        file_path = os.path.join(processed_data_path, file_name)
        joblib.dump((X, y), file_path)
        self.logger.info("Saved processed data at %s", file_path)
        mlflow.log_artifact(file_path)

    def process_all(self):
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment('NER_BiLSTM_CRF_processing')
        with mlflow.start_run():
            for split in ['train', 'val', 'test']:
                data_file = os.path.join(project_root, config['training']['processed_data_path'], f'{split}.pkl')
                data = pd.read_pickle(data_file)
                X, y = self.preprocess_data(data)
                self.save_processed_data(X, y, f'bilstm_{split}.pkl')
                mlflow.log_param(f"processed_{split}_size", len(X))
                self.logger.info("Processed %s data and saved successfully.", split)

if __name__ == "__main__":
    processor = BilstmCRFDataProcessor()
    processor.process_all()
