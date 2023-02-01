import numpy as np
from sklearn import tree
from joblib import dump


from data import dims, counts, hyperparameters

inp=np.stack([dims,counts],axis=1)


#train classifiers to find the best hyperparameters for a given input

for k,v in hyperparameters.items():
    model = tree.DecisionTreeClassifier(max_depth=2)
    model = model.fit(inp, v)
    dump(model, 'trees/' + k + '_model.joblib')
