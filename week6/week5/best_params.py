from sklearn.metrics import roc_auc_score
import numpy as np
import json
import glob
import os
import DEAN
from loaddata import loaddata

from data import  hyperparameters
from joblib import load


par = ['bag','lr', 'depth', 'batch', 'rounds']

avg_hyperparameters = {k: (int(np.mean(v)) if k != 'lr' else np.mean(v)) for k,v in hyperparameters.items()} #get average and median from _params.py
median_hyperparameters = {k: (int(np.median(v)) if k != 'lr' else np.median(v)) for k,v in hyperparameters.items()} #get average and median from _params.py
tree_hyperparameters = {}
alternative_param = {}





file_list = glob.glob(os.path.join('/global/datasets/test/', "*.npz"))

data = []
dataset_name = []

for file_path in file_list:
    data.append(np.load(file_path))
    dataset_name.append(file_path.split('/')[-1][:-4])


avg_params_results = {}
median_params_results = {}
tree_params_results = {}
trees_hyperparameters = {}

for i in range(len(data)):
    print('dataset ', i+1, 'out of ', len(data), '.......................')
    print('-'*1000)
    print('dataset name: ', dataset_name[i])


    (x_train0, y_train), (x_test0, y_test) = loaddata(dataset_name[i])

    print('shape of tx: ', x_test0.shape)
    if len(x_train0.shape)>2:
        x_train0=np.reshape(x_train0,(x_train0.shape[0],np.prod(x_train0.shape[1:])))
        x_test0 =np.reshape(x_test0 ,(x_test0.shape[0],np.prod(x_test0.shape[1:])))

    
    with open('../results/week5/true-results-' + dataset_name[i] + '.json', 'r') as f:
        alternative_param  = json.load(f)['params']

    try:
        model = DEAN.DEAN(**avg_hyperparameters)
        y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)
        au = roc_auc_score(y_true,y_score)
        avg_params_results[dataset_name[i]] = au
        print('average _parameters results for dataset: ', dataset_name[i])
        print('auc_score: ', au)
    
    except:
        model = DEAN.DEAN(**alternative_param)
        y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)
        au = roc_auc_score(y_true,y_score)
        avg_params_results[dataset_name[i]] = au
        print('ALTERNATIVE results for dataset: ', dataset_name[i])
        print('auc_score: ', au)



    try:
        model = DEAN.DEAN(**median_hyperparameters)
        y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)
        au = roc_auc_score(y_true,y_score)
        median_params_results[dataset_name[i]] = au
        print('median _parameters results for dataset: ', dataset_name[i])
        print('auc_score: ', au)

    except:
        model = DEAN.DEAN(**alternative_param)
        y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)
        au = roc_auc_score(y_true,y_score)
        avg_params_results[dataset_name[i]] = au
        print('ALTERNATIVE results for dataset: ', dataset_name[i])
        print('auc_score: ', au)



    samples = x_test0.shape[0]
    features = x_test0.shape[1]

    print('samples: ', samples, '  and features: ', features)
    inp= np.asarray([samples, features]).reshape(1, -1)
    
    


    for p in par:
        if p != 'lr':
            tree = load('trees/' + p + '_model.joblib')
            tree_hyperparameters[p] = int(tree.predict(inp))
        else:
            tree = load('trees/' + p + '_model.joblib')
            tree_hyperparameters[p] = tree.predict(inp)

    trees_hyperparameters[dataset_name[i]] = tree_hyperparameters

    try:
        model = DEAN.DEAN(**tree_hyperparameters)
        y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)
        au = roc_auc_score(y_true,y_score)
        tree_params_results[dataset_name[i]] = au
        print('regression _parameters results for dataset: ', dataset_name[i])
        print('auc_score: ', au)

    except:
        model = DEAN.DEAN(**alternative_param)
        y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)
        au = roc_auc_score(y_true,y_score)
        avg_params_results[dataset_name[i]] = au
        print('ALTERNATIVE results for dataset: ', dataset_name[i])
        print('auc_score: ', au)
    

    
best_params_results = {}
best_params_results['mean'] = {'score': avg_params_results, 'config': avg_hyperparameters}
best_params_results['median'] = {'score': median_params_results, 'config': avg_hyperparameters}
best_params_results['tree'] = {'score': tree_params_results, 'config': trees_hyperparameters}


print('saving..............................................')

with open('../results/week5/avg-parameters-results.json', 'w', encoding='utf-8') as f:
    json.dump(avg_params_results, f, ensure_ascii=False, indent=4)
    

with open('../results/week5/median-parameters-results.json', 'w', encoding='utf-8') as f:
    json.dump(median_params_results, f, ensure_ascii=False, indent=4)
    
    
with open('../results/week5/regression-parameters-results.json', 'w', encoding='utf-8') as f:
    json.dump(tree_params_results, f, ensure_ascii=False, indent=4)

with open('../results/week5/best-parameters-results.json', 'w', encoding='utf-8') as f:
    json.dump(best_params_results, f, ensure_ascii=False, indent=4)