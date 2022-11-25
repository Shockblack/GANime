#--------------------------------------------------------------
# File: model.py
#
# Programmer: Aiden Zelakiewicz (zelakiewicz.1@osu.edu)
#
# Dependencies: pytorch
#
# Description:
#   Contains the model for the convolutional Generative 
#   Adversarial Network. This might turn into a version
#   of the least squares GAN (lsGAN) by https://arxiv.org/abs/1611.04076.
#
# Revision History:
#   08-Nov-2022:  File Created
#   25-Nov-2022:  Finished Generator implementation
#--------------------------------------------------------------

import torch.nn as nn
from blocks import *

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # Input is (nc) x 64 x 64
            # Calculate output size with: (W-F+2P)/S + 1
            conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), # 32x32

            nn.LeakyReLU(0.2),

            DiscBlock(ndf, ndf*2, kernel_size=4, stride=2, padding=1), # 16x16
            DiscBlock(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1), # 8x8
            DiscBlock(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1), # 4x4

            conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=0), # 1x1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.discriminator(x)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # Input is Z
            GenBlock(nz, ngf*16), # 2x2
            GenBlock(ngf*16, ngf*8), # 4x4
            GenBlock(ngf*8, ngf*4), # 8x8
            GenBlock(ngf*4, ngf*2), # 16x16
            GenBlock(ngf*2, ngf), # 32x32
            GenBlock(ngf, nc), # 64x64
            nn.Tanh(),
            )
    
    def forward(self, x):
        return self.generator(x)
