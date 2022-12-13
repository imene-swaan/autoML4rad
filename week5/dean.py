from sklearn.metrics import roc_auc_score
from flaml import tune
import numpy as np
import time

from main import DEAN



#load data, and change the shape into (samples, features)
from loaddata import loaddata    
(x_train0, y_train), (x_test0, y_test) = loaddata()
if len(x_train0.shape)>2:
    x_train0=np.reshape(x_train0,(x_train0.shape[0],np.prod(x_train0.shape[1:])))
    x_test0 =np.reshape(x_test0 ,(x_test0.shape[0],np.prod(x_test0.shape[1:])))
   




def train_one(**hyper):
    model = DEAN(**hyper)
    model.fit(x,y)

    return roc_auc_score(ty, model.decision_function(tx))

def optimization(config: dict):
    t0 = time.time()
    auc = train_one(**config)
    t1 = time.time()

    return {'score': auc, 'evaluation_cost': t1-t0}

hyperparameters = {'lr': tune.uniform(lower = 0.01, upper = 0.1), 
                    'batch': tune.randint(lower = 10, upper = 200),
                    'depth': tune.randint(lower = 2, upper = 8),
                    'bag': tune.randint(lower = 2, upper = 21 ),
                    'rounds': tune.randint(lower = 10, upper = 100 )
                    }
