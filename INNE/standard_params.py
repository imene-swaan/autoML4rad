from sklearn.metrics import roc_auc_score
import numpy as np
import json
import glob
import os
from pyod.models.inne import INNE
from loaddata import loaddata



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


    (x, y), (tx, ty) = loaddata(dataset_name[i])



    model = INNE()
    model.fit(x)
    p = model.decision_function(tx)
    au = roc_auc_score(ty,p)

    standard_results[dataset_name[i]] = au


    print('default parameters results for dataset: ', dataset_name[i])
    print('auc_score: ', au)

    




print('saving..............................................')

with open('../results/week5/standard-parameters-results.json', 'w', encoding='utf-8') as f:
    json.dump(standard_results, f, ensure_ascii=False, indent=4)
    
    


