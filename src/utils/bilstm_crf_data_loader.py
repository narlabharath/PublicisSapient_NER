import sys
import os
import pandas as pd
import joblib
import yaml
import logging
import mlflow
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"File directory: {os.path.dirname(__file__)}")
print(f"Project root set to: {project_root}")

# Load Configuration
config_path = os.path.join(project_root, 'config/bilstm_crf_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].detach().clone(), self.y[idx].detach().clone()

class BilstmCRFDataLoader:
    def __init__(self):
        # Configure Logging
        log_file = os.path.join(project_root, config['logging']['log_dir'], 'bilstm_crf_data_loader.log')
        self.logger = logging.getLogger("bilstm_crf_data_loader")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def load_data(self, split):
        file_path = os.path.join(project_root, config['training']['processed_data_path'], f'bilstm_{split}.pkl')
        X, y = joblib.load(file_path)
        self.logger.info("Loaded %s data from %s", split, file_path)
        return X, y

    def create_dataloader(self, X, y, batch_size):
        dataset = NERDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=self.collate_fn)
        self.logger.info("Created DataLoader with batch size %d", batch_size)
        self.logger.info("DataLoader length: %d", len(dataloader))
        return dataloader

    def collate_fn(self, batch):
        sentences, tags = zip(*batch)
        sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
        tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
        return sentences_padded, tags_padded

if __name__ == "__main__":
    data_loader = BilstmCRFDataLoader()
    X_train, y_train = data_loader.load_data('train')
    train_loader = data_loader.create_dataloader(X_train, y_train, batch_size=32)
    for batch in train_loader:
        sentences, tags = batch
        print("Batch Sentences Shape:", sentences.shape)
        print("Batch Tags Shape:", tags.shape)
        break
