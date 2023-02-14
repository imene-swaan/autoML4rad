# DEAN
### Deep Ensemble Anomaly Detection


- Use "python3 true-parameters.py" to tune the model automatically using tune() from FLAML.

- Use "python3 standard-parameters.py" to train one model with default hyperparameter values (from paper) or your own.

- Use "python3 _parameters.py" to train multiple models on different datasets.

- Use "python3 main.py" to train a classifier/regressor to auto tune your hyperparameters.

- Use "python3 best-parameters.py" to auto train and tune multiple models on different datasets.

- Change loaddata.py to load a different dataset.


### -- **Default hyperparameters** --

- lr:float= 0.03,
- batch:int= 100,
- depth:int= 3,
- bag:int= 10,
- index:int=0,
- rounds:int= 100,
