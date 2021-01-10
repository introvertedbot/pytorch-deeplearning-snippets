# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:50:18 2021

@author: IntrovertedBot
"""



import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# Define hyperparameters
image_size = 784
hidden_dim = 400
latent_dim = 20
batch_size = 128
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='/',
                                          train=False,
                                          transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Create directory to save the reconstructed and sampled images (if directory not present)
sample_dir = 'results'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    


# VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(image_size, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, image_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2_mean(h)
        log_var = self.fc2_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(h))
        return out
    
    def forward(self, x):
        # x: (batch_size, 1, 28,28) --> (batch_size, 784)
        mu, logvar = self.encode(x.view(-1, image_size))
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# Define model and optimizer
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


