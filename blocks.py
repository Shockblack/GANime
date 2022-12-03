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
def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def GenBlock(in_plane, out_plane, bn=True):
    block = nn.Sequential()
    # Using upsample instead of transpose conv
    # Good discussion here: https://distill.pub/2016/deconv-checkerboard/
    block.add_module('upsample', nn.Upsample(scale_factor=2))
    block.add_module('conv', conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1, bias=False))
    if bn:
        block.add_module('bn', nn.BatchNorm2d(out_plane, 0.8))

    block.add_module('leaky_relu', nn.ReLU(True))
    return block

def DiscBlock(in_channels, out_channels, kernel_size, stride, padding, bn=True):

    block= nn.Sequential()
    block.add_module('conv', conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    block.add_module('leaky_relu', nn.LeakyReLU(0.2))
    block.add_module('dropout', nn.Dropout2d(0.25))
    if bn:
        block.add_module('bn', nn.BatchNorm2d(out_channels, 0.8))

    return block