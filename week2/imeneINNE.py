from pyod.models.inne import INNE
from sklearn.metrics import roc_auc_score
from flaml import tune
import numpy as np
import time
import json

#Create an instance of the model and fit it to the training data
f=np.load("/global/cardio.npz")
x,tx,ty=f["x"],f["tx"],f["ty"]

clf=INNE()
clf.fit(x)
p=clf.decision_function(tx)
auc=roc_auc_score(ty,p)
print('roc auc score before tuning: ', auc)



def train_one(**hyper):
    model = INNE(**hyper)
    model.fit(x)

    return roc_auc_score(ty, model.decision_function(tx))

def optimization(config: dict):
    t0 = time.time()
    auc = train_one(**config)
    t1 = time.time()

    l.append(config)
    t.append(t1-t0)
    a.append(auc)

    return {'score': auc, 'evaluation_cost': t1-t0}

hyperparameters = {'n_estimators': tune.randint(lower=50, upper=1000), 
                    'max_samples': tune.randint(lower = 1, upper = x.shape[0]),
                    'contamination': tune.quniform(lower =0.01, upper= 0.49, q = 0.01)
                    }

l = []
t = []
a = []
analysis = tune.run(
    optimization,  # the function to evaluate a config
    config= hyperparameters,  # the search space defined
    metric="score",
    mode="max",  # the optimization mode, "min" or "max"
    num_samples= 100,  # the maximal number of configs to try, -1 means infinite
    )

print('best model hyperparameters', analysis.best_config)  # the best config
print('best roc auc score is:  ', analysis.best_trial.last_result['score'])
print('time_s', analysis.best_trial.last_result['evaluation_cost'])  # the best trial's result'


with open('best_hyperparameters.json', 'w', encoding='utf-8') as f:
    json.dump(analysis.best_config, f, ensure_ascii=False, indent=4)

with open('best_auc_score.json', 'w', encoding='utf-8') as f:
    json.dump(analysis.best_trial.last_result, f, ensure_ascii=False, indent=4)