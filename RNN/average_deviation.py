"""
This script is used to compute the average and deviation of 
all the metrics collected in the script model_comparison_2.py
"""

import numpy as np

# Test dataset metrics for LSTM and Baseline Model across different seeds
lstm_test_metrics = {
    'accuracy': [0.8142, 0.8028, 0.8222, 0.7775, 0.8016],
    'loss': [0.4467, 0.4361, 0.4229, 0.4950, 0.4889],
    'f1_score': [0.8000, 0.7995, 0.8130, 0.7971, 0.7818]
}

baseline_test_metrics = {
    'accuracy': [0.7764, 0.7764, 0.7741, 0.7741, 0.7764],
    'loss': [0.4608, 0.4519, 0.4789, 0.4486, 0.4506],
    'f1_score': [0.7642, 0.7698, 0.7432, 0.7691, 0.7845]
}

# Calculating the average and standard deviation for each metric
def calc_avg_std(metrics):
    avg_std = {}
    for key in metrics:
        avg_std[key] = {
            'average': np.mean(metrics[key]),
            'std_dev': np.std(metrics[key])
        }
    return avg_std

# Results
lstm_results = calc_avg_std(lstm_test_metrics)
baseline_results = calc_avg_std(baseline_test_metrics)

print("lstm_results",lstm_results)
print("baseline_results", baseline_results)

