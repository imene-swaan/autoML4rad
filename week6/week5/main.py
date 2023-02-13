import numpy as np
from joblib import dump
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



from data import dims, counts, hyperparameters

inp=np.stack([dims,counts],axis=1)


#train classifiers to find the best hyperparameters for a given input

for k,v in hyperparameters.items():
    pipe = make_pipeline(StandardScaler(), HuberRegressor(max_iter = 200))
    model = pipe.fit(inp, v)
    dump(model, 'trees/' + k + '_model.joblib')
