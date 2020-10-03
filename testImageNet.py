import numpy as numpy
import torch
import torchvision

p = '/Volumes/share/data/ILSVRC2012'
#p = '/mnt/tlab-nas/share/data/ILSVRC2012'
#p = '/Users/takataka/Desktop/ILSVRC2012'

#dL = torchvision.datasets.ImageNet(p)
dV = torchvision.datasets.ImageNet(p, split='val')
print(dV)

print('# number of classes:', len(dV.classes))

for k, classname in enumerate(dV.classes):
    print(k, classname)

print(dV.classes[0])
print(dV.imgs[0])
print()
print(dV.classes[999])
print(dV.imgs[-1])


