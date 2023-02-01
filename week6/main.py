import numpy as np
from sklearn import tree
from joblib import dump


from data import dims, counts, hyperparameters

inp=np.stack([dims,counts],axis=1)


#train a classifier to find the best n_estimators for a given input
clf_bag = tree.DecisionTreeClassifier(max_depth=2)
clf_bag =clf_bag.fit(inp,bag)
dump(clf_bag, 'filename.joblib') 


for k,v in hyperparameters.items():
    model = tree.DecisionTreeClassifier(max_depth=2)
    model = model.fit(inp, v)
    dump(model, 'trees/' + k + '_model.joblib')
