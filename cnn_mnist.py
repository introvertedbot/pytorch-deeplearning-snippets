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

