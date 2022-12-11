from pyod.models.inne import INNE
from sklearn.metrics import roc_auc_score
from flaml import tune
import numpy as np
import time
import json
import glob
import os
import pandas as pd
import sys


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

sys.path.insert(0, '..')

from week3.week3_summary import mean_params, median_params


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

    print('default parameters results...................')
    print('auc_score: ', au)

    
    print('training with mean of hyperparameters ......')
    print(mean_params)

    auc = optimization(mean_params)
    print('auc_score ', auc)

    r.append(auc)
    h.append(mean_params)






    print('training with median of hyperparameters ......')
    print(median_params)

    auc = optimization(median_params)
    print('auc_score ', auc)

    r.append(auc)
    h.append(median_params)







    print('best......................')
    best_results = max(r)
    print(best_results)
    print('with config.........')

    best_parameters = h[r.index(max(r))]
    print(best_parameters)



    print('saving..............')
    with open('../results/week4/summarised_params_results/'+dataset_name[i]+'.json', 'w', encoding='utf-8') as f:
        json.dump(best_parameters, f, ensure_ascii=False, indent=4)
    
    


