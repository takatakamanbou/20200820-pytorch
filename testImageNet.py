import numpy as numpy
import torch
import torchvision

p = '/Volumes/share/data/ILSVRC2012'
#p = '/Users/takataka/Desktop/ILSVRC2012'

d = torchvision.datasets.ImageNet(p)