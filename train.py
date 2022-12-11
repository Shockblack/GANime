#--------------------------------------------------------------
# File: train.py
#
# Programmer: Aiden Zelakiewicz (zelakiewicz.1@osu.edu)
#
# Dependencies: pytorch
#
# Description:
#   Contains the training loop for the GAN.
#
# Revision History:
#   25-Nov-2022:  File Created
#--------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import *
from data_prep import get_dataloader

# Initializing Hyperparameters
batch_size = 128
epochs = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

nc = 3
nz = 100
ngf = 64
ndf = 64

num_workers = 4

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device: ", device)
train_loader = get_dataloader(batch_size, 'data/', num_workers=0)

# Loading the models
generator = Generator(nz, ngf, nc).to(device)
discriminator = Discriminator(nc, ndf).to(device)

# Initializing weights
generator.apply(weights_init_tut)
discriminator.apply(weights_init_tut)

# Setting up the optimizers and loss function
criterion = nn.BCELoss().to(device)

optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2)) # Discriminator optimizer
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2)) # Generator optimizer

# Keeping track of losses
G_losses = []
D_losses = []

# Noise vector for testing and visualization
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Begin training loop
for epoch in range(epochs):
    # Loop through batches
    for i, data in enumerate(tqdm(train_loader)):
        
        #------------------------------------#
        #-------TRAINING DISCRIMINATOR-------#
        #------------------------------------#

        # Set gradients to zero
        discriminator.zero_grad(set_to_none=True)

        # Train the discriminator on real data
        real = data.to(device)
        batch = real.size(0)
        label_true = torch.ones(batch, device=device)

        # Passing real data through discriminator
        output = discriminator(real).view(-1) # Output is (batch_size, 1, 1, 1), view makes it just (batch_size)
        errD_real = criterion(output, label_true)
        errD_real.backward() # Backpropagating error

        # Train the discriminator on fake data
        noise = torch.randn(batch, nz, 1, 1, device=device)
        label_fake = torch.zeros(batch, device=device)

        # Generating fake data and passing it through discriminator
        fake = generator(noise)
        output = discriminator(fake.detach()).view(-1)
        errD_fake = criterion(output, label_fake)
        errD_fake.backward()

        # Updating discriminator
        errD = errD_real + errD_fake
        optimizerD.step()

        G_losses.append(errD.item())

        #------------------------------------#
        #---------TRAINING GENERATOR---------#
        #------------------------------------#

        # Set gradients to zero
        generator.zero_grad(set_to_none=True)

        output = discriminator(fake).view(-1)
        errG = criterion(output, label_true)
        errG.backward()

        # Updating generator
        optimizerG.step()
        G_losses.append(errG.item())

    # Printing losses
    print(f"Epoch {epoch+1}/{epochs} | Discriminator Loss: {errD.item():.4f} | Generator Loss: {errG.item():.4f}")
    

    with torch.no_grad():

        fake = generator(fixed_noise).cpu().numpy()
        
        fig, axs = plt.subplots(8, 8, figsize=(8, 8))
        axs = axs.flatten()
        for img, ax in zip(fake, axs):
            ax.imshow(img.transpose(1, 2, 0), aspect='auto', interpolation='none')
            ax.axis('off')
        plt.tight_layout()

        plt.savefig('figs/epoch_{}.png'.format(epoch+1))
        plt.close()
