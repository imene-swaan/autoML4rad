from sklearn.metrics import roc_auc_score
import numpy as np
import json
import glob
import os
import DEAN
from loaddata import loaddata
from flaml import tune
import time



file_list = glob.glob(os.path.join('/global/datasets/', "*.npz"))

data = []
dataset_name = []

for file_path in file_list:
    data.append(np.load(file_path))
    dataset_name.append(file_path.split('/')[-1][:-4])


best_params_results = {}


def train_one(**hyper):
    model = DEAN.DEAN(**hyper)
    y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)

    return roc_auc_score(y_true,y_score)

def optimization(config: dict):

    t0 = time.time()
    auc = train_one(**config)
    t1 = time.time()

    return {'score': auc, 'evaluation_cost': t1-t0}





for i in range(len(data)):
    print('dataset ', i+1, 'out of ', len(data), '.......................')
    print('-'*1000)
    print('dataset name: ', dataset_name[i])


    (x_train0, y_train), (x_test0, y_test) = loaddata(dataset_name[i])
    if len(x_train0.shape)>2:
        x_train0=np.reshape(x_train0,(x_train0.shape[0],np.prod(x_train0.shape[1:])))
        x_test0 =np.reshape(x_test0 ,(x_test0.shape[0],np.prod(x_test0.shape[1:])))
    


    hyperparameters = {
                    'bag': tune.randint(lower =2, upper= x_train0.shape[1]),
                    'lr': tune.uniform(lower = 0.02, upper= 0.08),
                    'depth': tune.randint(lower = 2, upper = 6),
                    'batch': tune.randint(lower= 50, upper = 150),
                    'rounds': tune.randint(lower= 10, upper = 150)
                    }

    analysis = tune.run(
        optimization,  # the function to evaluate a config
        config= hyperparameters,  # the search space defined
        metric="score",
        mode="max",  # the optimization mode, "min" or "max"
        num_samples= 50
        )

    t = {}

    t['auc_score'] = analysis.best_trial.last_result['score']
    t['params'] = analysis.best_trial.last_result['config']
    t['evaluation_cost'] = analysis.best_trial.last_result['evaluation_cost']

    best_params_results[dataset_name[i]] = t


    print('best parameters results for dataset: ', dataset_name[i])
    print('auc_score: ', t['auc_score'])
    print('with params: ', t['params'])

    




print('saving..............................................')

with open('../results/week5/best-parameters-results.json', 'w', encoding='utf-8') as f:
    json.dump(best_params_results, f, ensure_ascii=False, indent=4)
    
    


