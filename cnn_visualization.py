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

image = Image.open('dog.jpg')
plt.imshow(image)

vgg = models.vgg16(pretrained=True)
vgg.classifier[-1]    #vgg.classifier[6]



CUDA = torch.cuda.is_available()
if CUDA:
    vgg = vgg.cuda()

#Apply the transforms on the image
image = transform(image)

print(image.shape)

#Add the batch size
image = image.unsqueeze(0)

#Wrap it up in a variable
image = Variable(image)

#Transfer it to the GPU
if CUDA:
    image = image.cuda()


print(image.shape)

output = vgg(image)

print(output.shape)

#Transfer the 2D Tensor to 1D
output = output.squeeze(0)

print(output.shape)

labels = json.load(open('imagenet_class_index.json'))

index = output.max(0)

print(index)

index = str(index[1][0].item())
label = labels[index][1]

print(label)



module_list = list(vgg.features.modules())



print(vgg.features)
print(module_list[0])
print(module_list[1])
print(module_list[2])
module_list

outputs = []
names = []
for layer in module_list[1:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))

for feature_map in outputs:
    print(feature_map.shape)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    #Convert the 3D Tensor to 2D. Sum the same element of every channel
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale/feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    #Convert the 3D Tensor to 2D. Sum the same element of every channel
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale/feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())





