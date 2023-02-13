import numpy as np
import json
from scipy import stats
import sys
sys.path.insert(0, '..')

from week4.data import hyperparameters


n_estimators = hyperparameters['n_estimators']
contamination = hyperparameters['contamination']
max_samples = hyperparameters['max_samples']

mean_params = {
    'n_estimators': int(np.round(np.mean(n_estimators))),
    'contamination': float(np.mean(contamination)),
    'max_samples': int(np.round(np.mean(max_samples)))}

median_params = {
    'n_estimators': int(np.round(np.median(n_estimators))),
    'contamination': float(np.median(contamination)),
    'max_samples': int(np.round(np.median(max_samples)))}


with open('../results/week3/descriptive_params/mean_params.json', 'w', encoding='utf-8') as f:
        json.dump(mean_params, f, ensure_ascii=False, indent=4)

with open('../results/week3/descriptive_params/median_params.json', 'w', encoding='utf-8') as f:
        json.dump(median_params, f, ensure_ascii=False, indent=4)
