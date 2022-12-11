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
    print('dataset ', i+1, 'out of ', len(data), '.......................')
    print('dataset name: ', dataset_name[i])
    x,tx,ty = data[i]['x'], data[i]['tx'], data[i]['ty']

    clf=INNE()
    clf.fit(x)
    p=clf.decision_function(tx)
    au=roc_auc_score(ty,p)


    r = []
    h = []

    r.append(au)
    h.append({'n_estimators': 200,
             'contamination': 0.1,
             'max_samples': min(8, x.shape[0])
            })

    print('default parameters results...................')
    print('auc_score: ', au)

    
    for j in range(40):

        print('training with hyperparameters ', j+1)

        hy = {'n_estimators': int(hyperparameters['n_estimators'][j]),
         'contamination': float(hyperparameters['contamination'][j]),
         'max_samples': int(hyperparameters['max_samples'][j])
         }
        print(hy)

        if hy in h:
            print('same parameter values.....skipping...')
            continue

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

    print('comparing with pyod params .....')
    c = list(filter(lambda x: not x <= r[0], r))
    print(len(c), ' model params gave better results than pyod')


    print('saving..............................................')
    with open('../results/week4/'+dataset_name[i]+'.json', 'w', encoding='utf-8') as f:
        json.dump(best_parameters, f, ensure_ascii=False, indent=4)
    
    


