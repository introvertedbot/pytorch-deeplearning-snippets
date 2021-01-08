# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:43:13 2021

@author: -
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms, utils
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import json

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

image = Image.open(r'C:\Users\IntrovertedBot\Downloads\dog.jpg')
plt.imshow(image)

vgg = models.vgg16(pretrained=True)
vgg.classifier[-1]    #vgg.classifier[6]



CUDA = torch.cuda.is_available()
if CUDA:
    vgg = vgg.cuda()

#Apply the transforms on the image
image = transform(image)

print(image.shape)
