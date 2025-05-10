import sys
import os
import pandas as pd
import joblib
import mlflow
import yaml
import logging
from collections import Counter

# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"File directory: {os.path.dirname(__file__)}")
print(f"Project root set to: {project_root}")

# Load Configuration
config_path = os.path.join(project_root, 'config/data_prep_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configure MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment("NER_Shared_Resources")

# Configure Logging
log_file = os.path.join(project_root, config['logging']['log_dir'], 'shared_resources.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
processed_data_path = os.path.join(project_root, config['data']['processed_data_path'])
vocab_path = os.path.join(processed_data_path, 'vocab.pkl')
tag_map_path = os.path.join(processed_data_path, 'tag_map.pkl')

class SharedResources:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.tag2idx = {"<UNK>": 0}
        self.idx2tag = {0: "<UNK>"}

    def data_generator(self, file):
        df = pd.read_pickle(os.path.join(processed_data_path, file))
        for _, row in df.iterrows():
            yield row['Word'], row['Tag']

    def build_vocab(self, train_file):
        word_counter = Counter()
        tag_counter = Counter()
        logging.info("Building vocab from training data: %s", train_file)
        for words, tags in self.data_generator(train_file):
            word_counter.update(words)
            tag_counter.update(tags)

        # Build word2idx and idx2word from training data only
        for idx, word in enumerate(word_counter.keys(), start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        # Build tag2idx and idx2tag from training data only
        for idx, tag in enumerate(tag_counter.keys(), start=1):
            self.tag2idx[tag] = idx
            self.idx2tag[idx] = tag

        logging.info("Vocabulary and tag mapping built from training data.")
        mlflow.log_param("vocab_size", len(self.word2idx))
        mlflow.log_param("num_tags", len(self.tag2idx))

    def save_resources(self):
        joblib.dump((self.word2idx, self.idx2word), vocab_path)
        joblib.dump((self.tag2idx, self.idx2tag), tag_map_path)
        mlflow.log_artifact(vocab_path)
        mlflow.log_artifact(tag_map_path)
        logging.info("Resources saved: %s, %s", vocab_path, tag_map_path)


def main():
    with mlflow.start_run():
        train_file = config['data']['train_file']
        resources = SharedResources()
        resources.build_vocab(train_file)
        resources.save_resources()
        logging.info("Vocabulary and tag mappings generated and saved from training data.")


if __name__ == "__main__":
    main()
