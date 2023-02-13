import os
import numpy as np
import json
import glob


file_list = glob.glob(os.path.join('/global/datasets/test/', "*.npz"))


def make_json_ending(file):
    return file.split('/')[-1][:-4]+'.json'

def load_hyperparameters(file):
    with open("../results/week4/summarised_params_results/"+file,"r") as f:
        return json.load(f)

def load_inputs(file):
    f=np.load(file)
    x=f["tx"]
    return x.shape[0],x.shape[1]

my_hyperparameters=["n_estimators","contamination", 'max_samples']
n_samples=[]
n_features=[]
hyperparameters={k:[] for k in my_hyperparameters}


for file in file_list:
    samples,features=load_inputs(file)
    n_samples.append(samples)
    n_features.append(features)
    hypers=load_hyperparameters(make_json_ending(file))
    for k in my_hyperparameters:
        hyperparameters[k].append(hypers[k])

counts_test =np.array(n_samples)
dims_test =np.array(n_features)
hyperparameters={k:np.array(v) for k,v in hyperparameters.items()}

