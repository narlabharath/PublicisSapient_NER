import sys
import os
# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"File directory: {os.path.dirname(__file__)}")
print(f"Project root set to: {project_root}")


import torch
import torch.optim as optim
import yaml
import joblib
import logging
import mlflow
from src.models.bilstm_crf_model import BiLSTM_CRF
from src.utils.bilstm_crf_data_loader import BilstmCRFDataLoader



# Load Configuration
config_path = os.path.join(project_root, 'config/bilstm_crf_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BilstmCRFTrainer:
    def __init__(self):
        # Configure Logging
        log_file = os.path.join(project_root, config['logging']['log_dir'], 'bilstm_crf_training.log')
        self.logger = logging.getLogger("bilstm_crf_training")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # DataLoader initialization
        self.data_loader = BilstmCRFDataLoader()
        self.train_loader = self.data_loader.create_dataloader(*self.data_loader.load_data('train'), batch_size=config['model']['batch_size'])

        
        # Load vocab and tag mappings dynamically
        vocab_path = os.path.join(project_root, config['training']['processed_data_path'], 'vocab.pkl')
        tag_map_path = os.path.join(project_root, config['training']['processed_data_path'], 'tag_map.pkl')

        # Load vocab and tag map files
        with open(vocab_path, 'rb') as f:
            word2idx, _ = joblib.load(f)
        with open(tag_map_path, 'rb') as f:
            tag2idx, _ = joblib.load(f)
        # Get vocab and tag size dynamically
        vocab_size = len(word2idx)
        num_tags = len(tag2idx)
        self.logger.info(f"Dynamic vocab size: {vocab_size}, number of tags: {num_tags}")

        # Initialize Model
        self.model = BiLSTM_CRF(vocab_size, num_tags, config['model']['embedding_dim'], config['model']['hidden_dim']).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['model']['learning_rate'])
        self.logger.info("Model initialized with vocab size %d and tag size %d", vocab_size, num_tags)

    def train(self):
        self.model.train()
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])

        with mlflow.start_run():
            for epoch in range(config['model']['epochs']):
                total_loss = 0
                for batch_idx, (sentences, tags) in enumerate(self.train_loader):
                    sentences, tags = sentences.to(device), tags.to(device)

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Forward pass and loss calculation
                    loss = self.model.neg_log_likelihood(sentences, tags)
                    
                    batch_loss = torch.mean(loss)  # Aggregate batch loss
                    total_loss += batch_loss.item()

                    # Calculate the batch loss
                    batch_loss = torch.mean(self.model.neg_log_likelihood(sentences, tags))  # Aggregate to scalar
                    total_loss += batch_loss.item()

                    # Backward pass
                    batch_loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['model']['max_grad_norm'])
                    self.optimizer.step()

                    # Logging
                    if batch_idx % config['logging']['log_interval'] == 0:
                        self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {batch_loss.item():.4f}")

                avg_loss = total_loss / len(self.train_loader)
                print(f"Epoch {epoch+1}/{config['model']['epochs']}, Avg Loss: {avg_loss:.4f}")
                self.logger.info(f"Epoch {epoch+1}/{config['model']['epochs']}, Avg Loss: {avg_loss:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Save the model
            model_save_path = os.path.join(project_root, config['deployment']['model_save_path'], 'bilstm_crf_model.pth')
            torch.save(self.model.state_dict(), model_save_path)
            self.logger.info("Model saved at %s", model_save_path)
            mlflow.log_artifact(model_save_path)

if __name__ == "__main__":
    trainer = BilstmCRFTrainer()
    trainer.train()
