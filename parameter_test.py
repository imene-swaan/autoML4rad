from pyod.models.inne import INNE
from sklearn.metrics import roc_auc_score
from flaml import tune
import numpy as np
import time
import json
import glob
import os
import pandas as pd

file_list = glob.glob(os.path.join('/global/datasets/test/', "*.npz"))

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
    auc = train_one(**config)

    return auc



from data import hyperparameters


for i in range(len(data)):

    print('dataset name: ', dataset_name[i])
    x,tx,ty = data[i]['x'], data[i]['tx'], data[i]['ty']

    clf=INNE()
    clf.fit(x)
    p=clf.decision_function(tx)
    au=roc_auc_score(ty,p)


    r = []
    h = []

    r.append(au)
    h.append({'n_estimators': 'default',
             'contamination': 'default',
             'max_samples': 'default'})

    print('default parameters results.....')
    print('auc_score: ', au)

    
    for j in range(40):

        print('training with hyperparameters ', j)

        hy = {'n_estimators': hyperparameters['n_estimators'][j],
         'contamination': hyperparameters['contamination'][j],
         'max_samples': hyperparameters['max_samples'][j]
         }
        print(hy)

        auc = optimization(hy)
        print('auc_score ', auc)

        r.append(auc)
        h.append(hy)

    print('best......................')
    best_results = max(r)
    print(best_results)
    print('with config.........')

    best_parameters = h[r.index(max(r))]
    print(best_parameters)


    print('saving..............')
    with open('results/test/'+dataset_name[i]+'.json', 'w', encoding='utf-8') as f:
        json.dump(best_parameters, f, ensure_ascii=False, indent=4)
    
    


