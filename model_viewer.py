#--------------------------------------------------------------
# Filename: model_viewer.py
#
# Programmer: Aiden Zelakiewicz (zelakiewicz.1@osu.edu)
#
# Dependencies: pytorch
#
# Description:
#    This file is used to view the model architecture using 
#    Tensorboard. This is useful for visualizing the structure
#    of the model.
#
# Revision History:
#    10-Dec-2022:  File Created
#--------------------------------------------------------------

from model import Discriminator, Generator
from torchsummary import summary

gen = Generator()
disc = Discriminator()


summary(gen, (100, 1, 1))
summary(disc, (3, 64, 64))