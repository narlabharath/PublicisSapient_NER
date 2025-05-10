import sys
import os
# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

import joblib
import yaml
import logging
from sklearn_crfsuite import metrics
from src.utils.crf_utils import load_and_preprocess, idx2tag



# Load Configuration
config_path = os.path.join(project_root, 'config/crf_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configure Logging
log_file = os.path.join(project_root, config['logging']['log_dir'], 'crf_manual_check.log')
logger = logging.getLogger("crf_manual_check")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load CRF Model
model_path = os.path.join(project_root, config['deployment']['model_save_path'], 'crf_model.joblib')
model = joblib.load(model_path)
logger.info("Loaded CRF model from %s", model_path)

# Load a small sample for debugging
X_sample, y_sample = load_and_preprocess('train.pkl')

# Select the first sentence for simplicity
sample_features = X_sample[0:1]
sample_labels = y_sample[0:1]
logger.info("Sample input features: %s", sample_features)
logger.info("Sample true labels: %s", sample_labels)

# Convert label indices to tag names
def decode_labels(label_indices):
    return [[idx2tag.get(idx, '<UNK>') for idx in seq] for seq in label_indices]

# Perform prediction
try:
    y_pred = model.predict(sample_features)
    logger.info("Raw predictions: %s", y_pred)

    # Convert true labels from indices to tag names
    y_true_tags = decode_labels(sample_labels)
    y_pred_tags = y_pred

    # Flatten the lists for F1 score calculation
    y_true_flat = [tag for seq in y_true_tags for tag in seq]
    y_pred_flat = [tag for seq in y_pred_tags for tag in seq]

    # Calculate F1 Score
    f1 = metrics.flat_f1_score([y_true_flat], [y_pred_flat], average='weighted')
    accuracy = metrics.flat_accuracy_score([y_true_flat], [y_pred_flat])
    report = metrics.flat_classification_report([y_true_flat], [y_pred_flat])

    # Print the results
    print("Sample True Tags:", y_true_tags)
    print("Predicted Tags:", y_pred_tags)
    print("F1 Score (Weighted):", f1)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    # Log the results
    logger.info("Sample True Tags: %s", y_true_tags)
    logger.info("Predicted Tags: %s", y_pred_tags)
    logger.info("F1 Score (Weighted): %f", f1)
    logger.info("Accuracy: %f", accuracy)
    logger.info("Classification Report:\n%s", report)

except Exception as e:
    logger.error("Error during manual prediction check: %s", str(e))
