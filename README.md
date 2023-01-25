# DEAN
### Deep Ensemble Anomaly Detection


- Use "python3 Tuning-DEAN.py" to tune the model automatically using tune() from FLAML.
- Use "python3 DEAN.py" class to train one model with default hyperparameter values (from paper) or your own.
- Change loaddata.py to load a different dataset.
- Find the best AUC score achieved with tuned hyperparameters values in best_auc_score.json.


### -- **Default hyperparameters** --

- lr:float= 0.03,
- batch:int= 100,
- depth:int= 3,
- bag:int= 10,
- index:int=0,
- rounds:int= 100,
