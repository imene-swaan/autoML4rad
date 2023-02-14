import os
import numpy as np
import json
import glob
from loaddata2 import loaddata
import cv2



f = open('../results/week5/parameters-results.json')
h = json.load(f)


par = ['n_estimators','max_samples']

n_samples=[]
n_features=[]
hyperparameters={k:[] for k in par}


for k,v in h.items():
    for p in par:
        hyperparameters[p].append(v['params'][p])
    

    (x_train0, y_train), (x_test0, y_test) = loaddata(k)
    samples = x_test0.shape[0]
    features = x_test0.shape[1]

    n_samples.append(samples)
    n_features.append(features)



counts =np.array(n_samples)
dims =np.array(n_features)

print(len(counts))
