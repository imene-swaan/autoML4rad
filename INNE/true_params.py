from sklearn.metrics import roc_auc_score
import numpy as np
import json
import glob
import os
from pyod.models.inne import INNE
from flaml import tune
import time



file_list = glob.glob(os.path.join('/global/datasets/test/', "*.npz"))

data = []
dataset_name = []

for file_path in file_list:
    data.append(np.load(file_path))
    dataset_name.append(file_path.split('/')[-1][:-4])


best_params_results = {}


def train_one(**hyper):
    model = INNE(**hyper)
    model.fit(x)

    return roc_auc_score(ty ,model.decision_function(tx))

def optimization(config: dict):

    t0 = time.time()
    auc = train_one(**config)
    t1 = time.time()

    return {'score': auc, 'evaluation_cost': t1-t0}


hyperparameters = {'n_estimators': tune.randint(lower=50, upper=500),
                    'max_samples': tune.randint(lower = 10, upper= 300)
                    }


for i in range(len(data)):
    print('dataset ', i+1, 'out of ', len(data), '.......................')
    print('-'*1000)
    print('dataset name: ', dataset_name[i])


    x,tx,ty = data[i]['x'], data[i]['tx'], data[i]['ty']


    samples, features = tx.shape

    print('dimentions: (s,f) ----', samples, features)

    
    a = 10
    
    print('number of trials a:   ', a)
    print('-'*1000)

    analysis = tune.run(
        optimization,  # the function to evaluate a config
        config= hyperparameters,  # the search space defined
        metric="score",
        mode="max",  # the optimization mode, "min" or "max"
        num_samples= a
        )

    t = {'dataset': dataset_name[i]}

    t['auc_score'] = analysis.best_trial.last_result['score']
    t['params'] = analysis.best_trial.last_result['config']
    t['evaluation_cost'] = analysis.best_trial.last_result['evaluation_cost']

    best_params_results[dataset_name[i]] = t


    print('best parameters results for dataset: ', dataset_name[i])
    print('auc_score: ', t['auc_score'])
    print('with params: ', t['params'])

    print('saving..............................................')

    with open('../results/week5/true-results-' + dataset_name[i] + '.json', 'w', encoding='utf-8') as f:
        json.dump(best_params_results[dataset_name[i]], f, ensure_ascii=False, indent=4)
    

with open('../results/week5/true-parameters-results.json', 'w', encoding='utf-8') as f:
        json.dump(best_params_results, f, ensure_ascii=False, indent=4)







