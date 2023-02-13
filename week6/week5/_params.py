from sklearn.metrics import roc_auc_score
import numpy as np
import json
import glob
import os
import DEAN
from loaddata2 import loaddata
from flaml import tune
import time


# get list of all datasets
file_list = glob.glob(os.path.join('/global/datasets/', "*.npz"))
dataset_name = []
for file_path in file_list:
    dataset_name.append(file_path.split('/')[-1][:-4])


# get list of already tuned datasets
file_done = sorted(glob.glob('../results/week5/parameters-results-*.json'))
dataset_done = []
dataset_name_done = []
for file_path in file_done:
    with open(file_path, 'r') as f:  
        dataset_done.append(json.load(f))
    dataset = file_path.split('/')[-1].split('-')[-1].split('.')[0]
    dataset_name_done.append(dataset)

print('number of datasets already done: ', len(dataset_name_done))

# remove datasets already tuned
dataset_name = list(filter(lambda x: not x in dataset_name_done, dataset_name))
print('number of datasets left: ', len(dataset_name))

# save results in dict
best_params_results = {}


# tuning functions
def train_one(**hyper):
    model = DEAN.DEAN(**hyper)
    y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)
    return roc_auc_score(y_true,y_score)

def optimization(config: dict):
    t0 = time.time()
    auc = train_one(**config)
    t1 = time.time()
    return {'score': auc, 'evaluation_cost': t1-t0}





for i in range(len(dataset_name)):
    print('dataset ', i+1, 'out of ', len(dataset_name), '.......................')
    print('-'*1000)
    print('dataset name: ', dataset_name[i])


    (x_train0, y_train), (x_test0, y_test) = loaddata(dataset_name[i])
    if len(x_train0.shape)>2:
        x_train0=np.reshape(x_train0,(x_train0.shape[0],np.prod(x_train0.shape[1:])))
        x_test0 =np.reshape(x_test0 ,(x_test0.shape[0],np.prod(x_test0.shape[1:])))
    

    samples, features = x_train0.shape
    print('dimentions: (s,f) ----', samples, features)
    
    a = 5

        

    hyperparameters = {
                    'bag': tune.randint(lower = 1, upper= x_train0.shape[1]-1),
                    'lr': tune.uniform(lower = 0.02, upper= 0.05),
                    'depth': tune.randint(lower = 2, upper = 4),
                    'batch': tune.randint(lower= 50, upper = 150),
                    'rounds': tune.randint(lower= 20, upper = 60)
                    }

    analysis = tune.run(
        optimization,  # the function to evaluate a config
        config= hyperparameters,  # the search space defined
        metric="score",
        mode="max",  # the optimization mode, "min" or "max"
        num_samples= a
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

    with open('../results/week5/parameters-results-' + dataset_name[i] + '.json', 'w', encoding='utf-8') as f:
        json.dump(best_params_results[dataset_name[i]], f, ensure_ascii=False, indent=4)
    

print('joining previous results with new results in one file..............................................')
for i in range(len(dataset_name_done)):
    best_params_results[dataset_name_done[i]] = dataset_done[i]

print('datasets tuned: ', len(best_params_results.keys()))


print('saving final file as parameters-results.json in results/week5/ ..............................................')

with open('../results/week5/parameters-results.json', 'w', encoding='utf-8') as f:
    json.dump(best_params_results, f, ensure_ascii=False, indent=4)
    
    


