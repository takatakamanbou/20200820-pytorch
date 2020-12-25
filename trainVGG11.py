import numpy as numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import datetime

import ilsvrc2012

# loading VGG11-bn pretrained model
vgg11_bn = torchvision.models.vgg11_bn(pretrained=True)
#vgg11_bn.eval()
print(vgg11_bn)

# device
use_gpu_if_available = True
use_CUDA = use_gpu_if_available and torch.cuda.is_available()
if use_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('# using', device)

nn = vgg11_bn.to(device)


# dataset & dataloader
dL = ilsvrc2012.datasetsL
bsize = 64
dl = torch.utils.data.DataLoader(dL, batch_size=bsize, shuffle=True, pin_memory=use_CUDA, num_workers=8)
nbatch = len(dl)

# loss & optimizer
#criterion = nn.

s1 = datetime.datetime.now()




with torch.no_grad():
    for ib, rv in enumerate(dl):
        if ib == N:
            break
        X, lab = rv[0].to(device), rv[1].to(device)
        output = nn(X)
        pred = output.max(1, keepdim=True)[1]
        nc = pred.eq(lab.view_as(pred)).sum().item()
        nd = len(X)
        sb = datetime.datetime.now()
        print(f'# {ib}  {nc}/{nd}')
        ncorrect += nc
        ntotal += nd

s2 = datetime.datetime.now()

nerror = ntotal - ncorrect
print(f'{nerror}/{ntotal} = {nerror/ntotal:.2f}')
print(f'{s2 - s1}')
