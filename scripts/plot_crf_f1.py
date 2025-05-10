import sys
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import logging

# Automatically set the project root as the base path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"File directory: {os.path.dirname(__file__)}")
print(f"Project root set to: {project_root}")    

# Load Configuration
config_path = os.path.join(project_root, 'config/crf_config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configure Logging
log_file = os.path.join(project_root, config['logging']['log_dir'], 'plot_crf_f1.log')
logger = logging.getLogger("plot_crf_f1")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load evaluation results
results_path = os.path.join(project_root, config['deployment']['inference_output_path'], 'crf_predictions.pkl')
y_pred = joblib.load(results_path)
logger.info("Loaded predictions from %s", results_path)

# Load F1 Scores
f1_scores_path = os.path.join(project_root, config['deployment']['inference_output_path'], 'crf_per_entity_f1.txt')
with open(f1_scores_path, 'r') as file:
    lines = file.readlines()

# Parse F1 Scores
def parse_f1_scores(lines):
    data = {}
    for line in lines:
        if line.startswith(" ") and "%" not in line:
            parts = line.strip().split()
            if len(parts) > 4:
                entity, precision, recall, f1, support = parts[:5]
                data[entity] = float(f1)
    return data

f1_scores = parse_f1_scores(lines)

# Plotting
def plot_f1_scores(f1_scores):
    entities = list(f1_scores.keys())
    scores = list(f1_scores.values())
    plt.figure(figsize=(12, 6))
    plt.bar(entities, scores, color='skyblue')
    plt.title("Per-Entity F1-Score for CRF Model")
    plt.xlabel("Entity Types")
    plt.ylabel("F1-Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(project_root, config['deployment']['inference_output_path'], 'crf_f1_plot.png')
    plt.savefig(save_path)
    plt.show()
    logger.info("F1-score plot saved at %s", save_path)

# Generate Plot
plot_f1_scores(f1_scores)
