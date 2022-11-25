#--------------------------------------------------------------
# Filename: blocks.py
#
# Programmer: Aiden Zelakiewicz (zelakiewicz.1@osu.edu)
#
# Dependencies: numpy
#
# Description:
#   Contains blocks to be used in the creation of a convolutional
#   Generative Adversarial Network (GAN).
#
# Revision History:
#   01-Nov-2022:  File Created
#   25-Nov-2022:  Updated GenBlock to use upsample
#--------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# Create the 2D convolutional block
def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

# Create the 2D transposed convolutional block
def convTranspose2D(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def GenBlock(in_plane, out_plane):
    block = nn.Sequential(
        # Using upsample instead of transpose conv
        # Good discussion here: https://distill.pub/2016/deconv-checkerboard/
        nn.Upsample(scale_factor=2),
        conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1),) # This should retain the same size
    return block

def DiscBlock(in_channels, out_channels, kernel_size, stride, padding):
    block= nn.Sequential(
            conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            )
    return block