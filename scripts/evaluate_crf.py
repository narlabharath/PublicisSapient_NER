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
from sklearn_crfsuite import metrics
import matplotlib.pyplot as plt
from src.utils.crf_utils import load_and_preprocess, idx2tag


# Load Configuration
config_path = os.path.join(project_root, 'config/crf_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configure Logging
log_file = os.path.join(project_root, config['logging']['log_dir'], 'evaluate_crf.log')
logger = logging.getLogger("evaluate_crf")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Configure MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

# Load CRF Model
model_path = os.path.join(project_root, config['deployment']['model_save_path'], 'crf_model.joblib')
model = joblib.load(model_path)
logger.info("Loaded CRF model from %s", model_path)

# Convert label indices to tag names
def decode_labels(label_indices):
    return [[idx2tag.get(idx, '<UNK>') for idx in seq] for seq in label_indices]

# Flatten label lists while filtering out empty sequences
def flatten_labels(label_lists):
    return [tag for seq in label_lists for tag in seq if tag not in ['<PAD>', '<UNK>']]

# Extract F1 scores from the classification report
def extract_entity_f1_scores(report):
    entity_scores = {}
    for line in report.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] not in ["accuracy", "weighted", "micro", "macro"]:
            entity_name = parts[0]
            f1_score = float(parts[3])
            entity_scores[entity_name] = f1_score
    return entity_scores

# Evaluation and Logging
def evaluate_and_log(X, y, dataset_name):
    y_pred = model.predict(X)

    # Decode true labels from indices to tag names
    y_true_tags = decode_labels(y)
    y_pred_tags = y_pred  # Raw predictions are already tag names

    # Flatten the lists for F1 score calculation
    y_true_flat = flatten_labels(y_true_tags)
    y_pred_flat = flatten_labels(y_pred_tags)

    # Calculate metrics
    f1 = metrics.flat_f1_score([y_true_flat], [y_pred_flat], average='weighted')
    accuracy = metrics.flat_accuracy_score([y_true_flat], [y_pred_flat])
    report = metrics.flat_classification_report([y_true_flat], [y_pred_flat])
    logger.info(f"{dataset_name} - F1 Score: {f1}, Accuracy: {accuracy}")
    mlflow.log_metric(f"{dataset_name}_f1_score", f1)
    mlflow.log_metric(f"{dataset_name}_accuracy", accuracy)
    return y_pred_tags, report, f1

# Plotting Function: Side-by-Side Bar Plot
def plot_f1_scores(f1_scores):
    plt.figure(figsize=(14, 7))

    # Extract entities and dataset names
    entities = list(next(iter(f1_scores.values())).keys())
    datasets = list(f1_scores.keys())
    x = range(len(entities))

    # Define bar width and positions
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]  # For Train, Validation, Test

    # Plot each dataset as a separate group
    for i, dataset_name in enumerate(datasets):
        scores = [f1_scores[dataset_name].get(entity, 0) for entity in entities]
        plt.bar([pos + offsets[i] for pos in x], scores, width=bar_width, label=dataset_name)

    # Adding title and labels
    plt.title("Per-Entity F1-Score Comparison (Train, Validation, Test)")
    plt.xlabel("Entity Types")
    plt.ylabel("F1-Score")
    plt.xticks([pos for pos in x], entities, rotation=45)
    plt.legend()

    # Save the plot
    plot_path = os.path.join(project_root, config['deployment']['inference_output_path'], 'crf_f1_scores.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    logger.info("F1-score plot saved at %s", plot_path)


# Main Evaluation
with mlflow.start_run():
    f1_scores = {}
    for dataset_name, file_name in zip(['Train', 'Validation', 'Test'], ['train.pkl', 'val.pkl', 'test.pkl']):
        X, y = load_and_preprocess(file_name)
        y_pred, report, f1 = evaluate_and_log(X, y, dataset_name)
        if len(y_pred) > 0 and any(len(seq) > 0 for seq in y_pred):
            entity_scores = extract_entity_f1_scores(report)
            f1_scores[dataset_name] = entity_scores
            # Save predictions
            pred_path = os.path.join(project_root, config['deployment']['inference_output_path'], f'crf_{dataset_name.lower()}_predictions.pkl')
            joblib.dump(y_pred, pred_path)
            mlflow.log_artifact(pred_path)
            logger.info("%s predictions saved at %s", dataset_name, pred_path)
    plot_f1_scores(f1_scores)
