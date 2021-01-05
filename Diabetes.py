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
from torch.utils.data import DataLoader

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

sc = StandardScaler()
x = sc.fit_transform(x)

x = torch.tensor(x)
y = torch.tensor(y).unsqueeze(1)

print(x.shape)
print(y.shape)

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)
    
dataset = Dataset(x, y)

train_loader = torch.utils.data.DataLoader(dataset = dataset,
                            batch_size=32,
                            shuffle=True)
    
# Let's have a look at the data loader
print(f'There is {len(train_loader)} batches in the dataset')
for x,y in train_loader:
    print("For one iteration (batch), there is:")
    print(f"Data:    {x.shape}")
    print(f"Labels:    {y.shape}")
    break


