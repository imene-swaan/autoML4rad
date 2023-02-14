import numpy as np
from sklearn import tree
from joblib import dump
from sklearn.linear_model import LinearRegression


from data import dims, counts, hyperparameters

inp=np.stack([dims,counts],axis=1)


#train classifiers to find the best hyperparameters for a given input

for k,v in hyperparameters.items():
    if k != 'lr':
        model = tree.DecisionTreeClassifier(max_depth=2)
        model.fit(inp, v)
        dump(model, 'trees/' + k + '_model.joblib')
    else:
        model = LinearRegression().fit(inp, v)
        dump(model, 'trees/' + k + '_model.joblib')
