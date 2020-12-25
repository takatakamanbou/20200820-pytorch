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
bsize = 64
dl = torch.utils.data.DataLoader(dL, batch_size=bsize, shuffle=True, pin_memory=use_CUDA, num_workers=8)
nbatch = len(dl)

# optimizer & criterion
#optimizer = optim.Adam(nn.parameters(), )
optimizer = optim.SGD(nn.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

s1 = datetime.datetime.now()

nn.train()

ncList = np.zeros(nbatch, dtype=int)
lossList = np.empty(nbatch)

for ib, rv in enumerate(dl):
    X, lab = rv[0].to(device), rv[1].to(device)
    optimizer.zero_grad()
    output = F.log_softmax(nn(X))
    loss = criterion(output, lab)
    loss.backward()
    optimizer.step()
    lossList[ib] = loss.item()
    pred = output.max(1, keepdim=True)[1]
    ncList[ib] = pred.eq(lab.view_as(pred)).sum().item()

    print(ib, ncList[ib], lossList[ib])


s2 = datetime.datetime.now()

