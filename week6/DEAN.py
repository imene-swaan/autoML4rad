import numpy as np
import DEAN
from flaml import tune
import time
import json
from sklearn.metrics import roc_auc_score


#load data, and change the shape into (samples, features)
from loaddata import loaddata    
(x_train0, y_train), (x_test0, y_test) = loaddata()
if len(x_train0.shape)>2:
    x_train0=np.reshape(x_train0,(x_train0.shape[0],np.prod(x_train0.shape[1:])))
    x_test0 =np.reshape(x_test0 ,(x_test0.shape[0],np.prod(x_test0.shape[1:])))


def train_one(**hyper):
    model = DEAN.DEAN(**hyper)
    y_true,y_score = model.fit(x_train0, y_train, x_test0, y_test)

    return roc_auc_score(y_true,y_score)

def optimization(config: dict):

    print('IM TRAINING \n------------------------\n------------------------\n------------------------\n------------------------\n------------------------\n------------------------\n------------------------\n------------------------\n------------------------\n------------------------')

    t0 = time.time()
    auc = train_one(**config)
    t1 = time.time()

    return {'score': auc, 'evaluation_cost': t1-t0}

#(index, bag, lr, depth, batch, rounds, pwr)

hyperparameters = {
                    'bag': tune.randint(lower =2, upper= x_train0.shape[1]-1),
                    'lr': tune.uniform(lower = 0.02, upper= 0.08),
                    'depth': tune.randint(lower = 2, upper = 6),
                    'batch': tune.randint(lower= 50, upper = 150),
                    'rounds': tune.randint(lower= 10, upper = 100)
                    }



analysis = tune.run(
    optimization,  # the function to evaluate a config
    config= hyperparameters,  # the search space defined
    metric="score",
    mode="max",  # the optimization mode, "min" or "max"
    num_samples= 200
    )


print('best model hyperparameters', analysis.best_config)  # the best config
print('best roc auc score is:  ', analysis.best_trial.last_result['score'])
print('time_s', analysis.best_trial.last_result['evaluation_cost'])  # the best trial's result'


with open('best_hyperparameters.json', 'w', encoding='utf-8') as f:
    json.dump(analysis.best_config, f, ensure_ascii=False, indent=4)

with open('best_auc_score.json', 'w', encoding='utf-8') as f:
    json.dump(analysis.best_trial.last_result, f, ensure_ascii=False, indent=4)
