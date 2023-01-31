from sklearn.metrics import roc_auc_score
import numpy as np
import json
import glob
import os
import DEAN
from loaddata import loaddata



f = open('../results/week5/parameters-results.json')
h = json.load(f)

par = ['bag','lr', 'depth', 'batch', 'rounds']

hy = {}

for p in par:
    hy[p] = []

for k,v in h.items():
    for p in par:
        hy[p].append(v['params'][p])



avg_hyperparameters = {
                    'bag': np.mean(hy['bag']),
                    'lr': np.mean(hy['lr']),
                    'depth': np.mean(hy['depth']),
                    'batch': np.mean(hy['batch']),
                    'rounds': np.mean(hy['rounds'])
                    } #get average and median from _params.py



file_list = glob.glob(os.path.join('/global/datasets/test/', "*.npz"))

data = []
dataset_name = []

for file_path in file_list:
    data.append(np.load(file_path))
    dataset_name.append(file_path.split('/')[-1][:-4])


standard_results = {}

for i in range(len(data)):
    print('dataset ', i+1, 'out of ', len(data), '.......................')
    print('-'*1000)
    print('dataset name: ', dataset_name[i])


    (x_train0, y_train), (x_test0, y_test) = loaddata(dataset_name[i])
    if len(x_train0.shape)>2:
        x_train0=np.reshape(x_train0,(x_train0.shape[0],np.prod(x_train0.shape[1:])))
        x_test0 =np.reshape(x_test0 ,(x_test0.shape[0],np.prod(x_test0.shape[1:])))
    



    model = DEAN.DEAN(**avg_hyperparameters)

    y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)

    au = roc_auc_score(y_true,y_score)

    standard_results[dataset_name[i]] = au


    print('default parameters results for dataset: ', dataset_name[i])
    print('auc_score: ', au)

    




print('saving..............................................')

with open('../results/week5/avg-parameters-results.json', 'w', encoding='utf-8') as f:
    json.dump(standard_results, f, ensure_ascii=False, indent=4)
    
    


