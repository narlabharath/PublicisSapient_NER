import sys
import os
import pandas as pd
import joblib
import logging

# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"File directory: {os.path.dirname(__file__)}")
print(f"Project root set to: {project_root}")

# Load vocab and tag mappings
processed_data_path = os.path.join(project_root, 'data/processed')
vocab_path = os.path.join(processed_data_path, 'vocab.pkl')
tag_map_path = os.path.join(processed_data_path, 'tag_map.pkl')

with open(vocab_path, 'rb') as f:
    word2idx, idx2word = joblib.load(f)
with open(tag_map_path, 'rb') as f:
    tag2idx, idx2tag = joblib.load(f)

# CRF-specific feature extraction function
def word2features(sent, i):
    word = sent[i]
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

# Process the data for CRF input
def process_data(data):
    X = [[word2features(sent, i) for i in range(len(sent))] for sent in data['Word']]
    y = [[tag2idx.get(tag, tag2idx['<UNK>']) for tag in tags] for tags in data['Tag']]
    return X, y

# Load and preprocess data
def load_and_preprocess(file_name):
    data_path = os.path.join(processed_data_path, file_name)
    data = pd.read_pickle(data_path)
    return process_data(data)

# Convert predictions from indices to tag names
def decode_predictions(predictions):
    return [[idx2tag.get(idx, '<UNK>') for idx in seq] for seq in predictions]
