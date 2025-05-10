import sys
import os
import pandas as pd
import joblib
import mlflow
import yaml
import logging

# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"File directory: {os.path.dirname(__file__)}")
print(f"Project root set to: {project_root}")

# Load Configuration
config_path = os.path.join(project_root, 'config/crf_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configure MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment("NER_CRF_Preprocessing")

# Configure Logging
log_file = os.path.join(project_root, config['logging']['log_dir'], 'crf_preprocessing.log')
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

# Load vocab and tags
with open(vocab_path, 'rb') as f:
    word2idx, idx2word = joblib.load(f)
with open(tag_map_path, 'rb') as f:
    tag2idx, idx2tag = joblib.load(f)


# Feature extraction function
def word2features(sent, i):
    word = sent[i]
    # Handle missing or non-string words
    if not isinstance(word, str):
        word = "<UNK>"
    features = {
        'word': word,
        'is_capitalized': word[0].isupper(),
        'is_numeric': word.isdigit(),
        'prefix-1': word[:1],
        'prefix-2': word[:2],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:],
    }
    if i > 0:
        prev_word = sent[i - 1]
        if not isinstance(prev_word, str):
            prev_word = "<UNK>"
        features['prev_word'] = prev_word
    else:
        features['prev_word'] = '<BOS>'
    if i < len(sent) - 1:
        next_word = sent[i + 1]
        if not isinstance(next_word, str):
            next_word = "<UNK>"
        features['next_word'] = next_word
    else:
        features['next_word'] = '<EOS>'
    return features

# Process data for CRF
def process_sentence(sentence):
    return [word2features(sentence, i) for i in range(len(sentence))]

# Main function
def main():
    with mlflow.start_run():
        train_path = os.path.join(processed_data_path, config['data']['train_file'])
        train_data = pd.read_pickle(train_path)

        X_train = [process_sentence(words) for words in train_data['Word']]
        # y_train = [[tag2idx.get(tag, tag2idx['<UNK>']) for tag in tags] for tags in train_data['Tag']]
        y_train = [tags for tags in train_data['Tag']]

        # Save processed data
        joblib.dump((X_train, y_train), os.path.join(processed_data_path, 'crf_train.pkl'))
        logging.info("CRF preprocessing completed. Training data saved.")
        mlflow.log_param("train_data_size", len(X_train))
        mlflow.log_artifact(log_file)

if __name__ == "__main__":
    main()
