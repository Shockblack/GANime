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
#   06-Dec-2022:  Replaced generator with convTranspose2d
#--------------------------------------------------------------

import torch.nn as nn
from blocks import *

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            # Calculate output size with: (W-F+2P)/S + 1
            DiscBlock(nc, ndf, 4, 2, 1, bn=False), # 32x32
            DiscBlock(ndf, ndf*2, 4, 2, 1), # 16x16
            DiscBlock(ndf*2, ndf*4, 4, 2, 1), # 8x8
            DiscBlock(ndf*4, ndf*8, 4, 2, 1), # 4x4

            conv2d(ndf*8, 1, 4, 1, 0, bias=False), # 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            GenBlock(nz, ngf*8, 4, 1, 0),
            # State size. (ngf*8) x 4 x 4
            GenBlock(ngf*8, ngf*4, 4, 2, 1),
            # State size. (ngf*4) x 8 x 8
            GenBlock(ngf*4, ngf*2, 4, 2, 1),
            # State size. (ngf*2) x 16 x 16
            GenBlock(ngf*2, ngf, 4, 2, 1),
            # State size. (ngf) x 32 x 32
            convTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # State size. (nc) x 64 x 64
        )
    def forward(self, x):
        return self.main(x)


def weights_init(Layer):
    """
    Initializes the weights of the layer, w.
    """
    name = Layer.__class__.__name__
    if name == 'conv':
        nn.init.kaiming_normal_(Layer.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif name == 'bn':
        nn.init.normal_(Layer.weight.data, 1.0, 0.02)
        nn.init.constant_(Layer.bias.data, 0)
        
def weights_init_tut(Layer):
    classname = Layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(Layer.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(Layer.weight.data, 1.0, 0.02)
        nn.init.constant_(Layer.bias.data, 0)