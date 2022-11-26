from pyod.models.inne import INNE
from sklearn.metrics import roc_auc_score
from flaml import tune
import numpy as np
import time
import json
import glob
import os
import pandas as pd

file_list = glob.glob(os.path.join('/global/datasets/', "*.npz"))

data = []
dataset_name = []

for file_path in file_list:
    data.append(np.load(file_path))
    dataset_name.append(file_path.split('/')[-1][:-4])


def train_one(**hyper):
    model = INNE(**hyper)
    model.fit(x)

    return roc_auc_score(ty, model.decision_function(tx))

def optimization(config: dict):
    t0 = time.time()
    auc = train_one(**config)
    t1 = time.time()

    return {'score': auc, 'evaluation_cost': t1-t0}

hyperparameters = {'n_estimators': tune.randint(lower=50, upper=500), 
                    'contamination': tune.quniform(lower =0.01, upper= 0.40, q = 0.01),
                    'max_samples': tune.randint(lower = 10, upper= 300)
                    }


results = {}

for i in range(len(data)):
    x,tx,ty = data[i]['x'], data[i]['tx'], data[i]['ty']

    analysis = tune.run(
    optimization,  # the function to evaluate a config
    config= hyperparameters,  # the search space defined
    metric="score",
    mode="max",  # the optimization mode, "min" or "max"
    resources_per_trial={'cpu':4, 'gpu':0},
    num_samples= 10
    )
    f = ' '.join(['model', str(i+1), 'for dataset:', file_list[i]])


    with open('results/'+dataset_name[i]+'.json', 'w', encoding='utf-8') as f:
        json.dump(analysis.best_config, f, ensure_ascii=False, indent=4)


    #results[f] = [analysis.best_config,analysis.best_trial.last_result]
