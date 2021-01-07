# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:32:54 2021

@author: IntrovertedBot
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

mean_gray = 0.1307
stddev_gray = 0.3081

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((mean_gray,), (stddev_gray))])

train_dataset = datasets.MNIST(root = 'D:\\',
                               train = True,
                               transform = transforms,
                               download = True)

test_dataset = datasets.MNIST(root = 'D:\\',
                               train = False,
                               transform = transforms,
                               download = True)

random_img = train_dataset[20][0].numpy() * stddev_gray + mean_gray
plt.imshow(random_img.reshape(28,28), cmap = 'gray')

batch_size = 100
train_load = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size = batch_size,
                                         shuffle = True)
test_load = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Same padding --> input_size = output_size
        # Same padding = (filter_size - 1) / 2 --> (3 - 1)/2 = 1
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        # the output size of each of the 8 feature maps:
        # [(input_size - filtersize + 2(padding))/stride + 1] = ( 28 - 3 + 2 )1 + 1 = 28
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        # the output size = 28/2 = 14
        # Same padding = (5 - 1)/2 = 2
        self.cnn2 = nn.Conv2d(in_channels = 8, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
        # output size of each of the 32 feature maps [(14-5 + 2(2)/1 + 1)] = 14
        self.batchnorm2 = nn.BatchNorm2d(32)
        # Flatten the 32 feature maps: 7*7*32 = 1568
        self.fc1 = nn.Linear(1568, 600)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(600, 10)
        
    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.cnn2(out)
        out = self.batchnorm2(32)
        out = self.relu(out)
        out = self.maxpool(out)
        # Flatten the 32 feature maps from max pool to feed it to FC1(100, 1568)
        out = self.view(-1, 1568)
        # then we forward through our fully connected layer
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
        