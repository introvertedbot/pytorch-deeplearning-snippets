# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:01:26 2021

@author: IntrovertedBot
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

data = pd.read_csv(r'E:\The-Complete-Neural-Networks-Bootcamp-Theory-Applications-master\The-Complete-Neural-Networks-Bootcamp-Theory-Applications-master\diabetes.csv')

x = data.iloc[:,0:-1].values
y_string = list(data.iloc[:,-1])

y_int = []
for string in y_string:
    if string == 'positive':
        y_int.append(1)
    else:
        y_int.append(0)

y = np.array(y_int, dtype= 'float64')




