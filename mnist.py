# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:06:19 2021

@author: IntrovertedBot
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

input_size = 784
hidden_size = 400
out_size = 10
epochs = 10
batch_size = 100
learning_rate = 0.001

train_dataset = datasets.MNIST(root = 'D:\\',
                               train = True,
                               transform = transforms.ToTensor(),
                               download = True)
test_dataset = datasets.MNIST(root = 'D:\\',
                               train = False,
                               transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)


