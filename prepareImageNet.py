import numpy as numpy
import torch
import torchvision

#p = '/Volumes/share/data/ILSVRC2012'
#p = '/mnt/tlab-nas/share/data/ILSVRC2012'
p = '/Users/takataka/Desktop/ILSVRC2012'

dL = torchvision.datasets.ImageNet(p)
dV = torchvision.datasets.ImageNet(p, split='val')