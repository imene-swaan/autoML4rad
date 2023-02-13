from pyod.models.inne import INNE
from sklearn.metrics import roc_auc_score
from flaml import tune
import numpy as np
import time
import json
import glob
import os
import pandas as pd

file_list = glob.glob(os.path.join('../results/week4/', "*.npz"))

data = []
dataset_name = []

for file_path in file_list:
    data.append(np.load(file_path))
    dataset_name.append(file_path.split('/')[-1][:-4])