from sklearn.metrics import roc_auc_score
import numpy as np
import json
import glob
import os
from pyod.models.inne import INNE
from loaddata import loaddata
from data import  hyperparameters


par = ['n_estimators','max_samples']

avg_hyperparameters = {k: (int(np.mean(v)) if k != 'lr' else np.mean(v)) for k,v in hyperparameters.items()} #get average and median from _params.py
median_hyperparameters = {k: (int(np.median(v)) if k != 'lr' else np.median(v)) for k,v in hyperparameters.items()} #get average and median from _params.py





file_list = glob.glob(os.path.join('/global/datasets/test/', "*.npz"))
dataset_name = []
for file_path in file_list:
    dataset_name.append(file_path.split('/')[-1][:-4])

    

avg_params_results = {}
median_params_results = {}

for i in range(len(data)):
    print('dataset ', i+1, 'out of ', len(data), '.......................')
    print('-'*1000)
    print('dataset name: ', dataset_name[i])


    (x, y), (tx, ty) = loaddata(dataset_name[i])

    print('shape of tx: ', tx.shape)
    

    try:
        model = INNE(**avg_hyperparameters)
        model.fit(x)
        p = model.decision_function(tx)
        au = roc_auc_score(ty,p)
        avg_params_results[dataset_name[i]] = au
        print('average _parameters results for dataset: ', dataset_name[i])
        print('auc_score: ', au)
        
        with open('../results/week5/avg-parameters-results-'+ dataset_name[i] + '.json', 'w', encoding='utf-8') as f:
            json.dump({'auc score': au}, f, ensure_ascii=False, indent=4)
    
    except:
        print('ALTERNATIVE results for dataset: ', dataset_name[i], ' is true param')

    



    try:
        model = INNE(**median_hyperparameters)
        model.fit(x)
        p = model.decision_function(tx)
        au = roc_auc_score(ty,p)
        median_params_results[dataset_name[i]] = au
        print('median _parameters results for dataset: ', dataset_name[i])
        print('auc_score: ', au)
        
        with open('../results/week5/median-parameters-results-'+ dataset_name[i] + '.json', 'w', encoding='utf-8') as f:
            json.dump({'auc score': au}, f, ensure_ascii=False, indent=4)

    except:
        print('ALTERNATIVE results for dataset: ', dataset_name[i], ' is true param')

   
    

    
best_params_results = {}
best_params_results['mean'] = {'score': avg_params_results, 'config': avg_hyperparameters}
best_params_results['median'] = {'score': median_params_results, 'config': avg_hyperparameters}


print('saving..............................................')
with open('../results/week5/best-parameters-results.json', 'w', encoding='utf-8') as f:
    json.dump(best_params_results, f, ensure_ascii=False, indent=4)
