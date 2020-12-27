import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import datetime
import sys

import ilsvrc2012
import DCTnet


# loading VGG16 pretrained model
#vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16 = DCTnet.DCTnet()
print(vgg16)


# device
use_gpu_if_available = True
use_CUDA = use_gpu_if_available and torch.cuda.is_available()
if use_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('# using', device)

nn = vgg16.to(device)


# dataset & dataloader
dL = ilsvrc2012.datasets('L')
bsize = 256
dl = torch.utils.data.DataLoader(dL, batch_size=bsize, shuffle=True, pin_memory=use_CUDA, num_workers=16)
nbatch = len(dl)

# optimizer
#optimizer = optim.SGD(nn.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(nn.parameters(), lr=0.0001, weight_decay=0.0)


nn.train()

nb = 1024 // bsize
ncList = np.zeros(nb, dtype=int)
lossList = np.zeros(nb)

s1 = datetime.datetime.now()

for ib, rv in enumerate(dl):
    X, lab = rv[0].to(device), rv[1].to(device)
    optimizer.zero_grad()
    output = F.log_softmax(nn(X), dim=1)
    loss = F.nll_loss(output, lab)
    loss.backward()
    optimizer.step()

    lossList[ib % nb] = loss.sum().item()
    pred = output.max(1, keepdim=True)[1]
    ncList[ib % nb] = pred.eq(lab.view_as(pred)).sum().item()

    if ib % nb == nb - 1:
        nc = np.sum(ncList)
        loss_mean = np.mean(lossList)/bsize
        print(f'{ib}/{nbatch}  {loss_mean:.6f}  {nc}/{bsize*nb} = {nc/(bsize*nb)}')
        sys.stdout.flush()

    #if ib == 100:
    #    break

s2 = datetime.datetime.now()

print(s2-s1)

fnParam = 'data/ex20201228_trainDCT_epoch001.pth'
with open(fnParam, mode='wb') as f:
    torch.save(nn.state_dict(), f)    


