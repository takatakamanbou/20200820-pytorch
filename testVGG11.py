import numpy as numpy
import torch
import torch.nn.functional as F
import torchvision
import datetime

import ilsvrc2012

# loading VGG16 model
vgg11_bn = torchvision.models.vgg11_bn(pretrained=False)
print(vgg11_bn)

# loading the parameters
fnParam = 'data/hoge.pth'
with open(fnParam, mode='rb') as f:
    vgg11_bn.load_state_dict(torch.load(f))

# device
use_gpu_if_available = True
use_CUDA = use_gpu_if_available and torch.cuda.is_available()
if use_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('# using', device)

nn = vgg11_bn.to(device)
nn.eval()

# dataset & dataloader
dT = ilsvrc2012.datasetsT
bsize = 256
dl = torch.utils.data.DataLoader(dT, batch_size=bsize, shuffle=False, pin_memory=use_CUDA, num_workers=16)
nbatch = len(dl)

s1 = datetime.datetime.now()

N = 100  # number of batch to be fed

ncorrect = 0
ntotal = 0
with torch.no_grad():
    for ib, rv in enumerate(dl):
        if ib == N:
            break
        X, lab = rv[0].to(device), rv[1].to(device)
        output = F.log_softmax(nn(X), dim=1)
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
