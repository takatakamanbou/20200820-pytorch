import numpy as numpy
import torch
import torchvision
import datetime

# path to ILSVRC2012
p = '/mnt/data/ILSVRC2012'

# loading VGG16 pretrained model
vgg16 = torchvision.models.vgg16(pretrained=True)

# setting the image transformation
#    cf. https://pytorch.org/docs/stable/torchvision/models.html
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.CenterCrop((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




s0 = datetime.datetime.now()

# dataset
dV = torchvision.datasets.ImageNet(p, split='val', transform=trans)
print(dV)

s1 = datetime.datetime.now()
print('# datasets initialization:', s1 - s0)

bsize = 10
dl = torch.utils.data.DataLoader(dV, batch_size=bsize, shuffle=True)

s2 = datetime.datetime.now()
print('# dataloader initialization:', s2 - s1)

#X, lab = next(iter(dl))
#print(X.shape)

#X, lab = next(iter(dl))
#print(X.shape)

nbatch = len(dl)

s3 = datetime.datetime.now()

N = 10

vgg16.eval()
with torch.no_grad():
    for ib, rv in enumerate(dl):
        if ib == N:
            break
        X, lab = rv
        output = vgg16(X)
        pred = output.max(1, keepdim=True)[1]
        ncorrect = pred.eq(lab.view_as(pred)).sum().item()
        print(ncorrect)
        sb = datetime.datetime.now()
        print('#', ib, X.shape, lab)

s4 = datetime.datetime.now()
print(f'# loading {N} batches (batchsize = {bsize}):', s4 - s3)
