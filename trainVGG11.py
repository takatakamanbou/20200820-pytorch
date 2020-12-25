import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import datetime

import ilsvrc2012

# loading VGG11-bn pretrained model
vgg11_bn = torchvision.models.vgg11_bn(pretrained=False)
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
bsize = 256
dl = torch.utils.data.DataLoader(dL, batch_size=bsize, shuffle=True, pin_memory=use_CUDA, num_workers=16)
nbatch = len(dl)

# optimizer
#optimizer = optim.SGD(nn.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(nn.parameters(), lr=0.001, weight_decay=0.0)


nn.train()

nb = 1024 // bsize
ncList = np.zeros(nb, dtype=int)
lossList = np.zeros(nb)

s1 = datetime.datetime.now()

for ib, rv in enumerate(dl):
    X, lab = rv[0].to(device), rv[1].to(device)
    optimizer.zero_grad()
    output = F.log_softmax(nn(X))
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

    #if ib == 100:
    #    break

s2 = datetime.datetime.now()

print(s2-s1)

fnParam = 'data/hoge.pth'
with open(fnParam, mode='wb') as f:
    torch.save(nn.state_dict(), f)    


