import numpy as numpy
import torch
import torchvision
import datetime

# path to ILSVRC2012
p = '/mnt/data/ILSVRC2012'

# loading VGG11-bn pretrained model
vgg11_bn = torchvision.models.vgg11_bn(pretrained=True)
vgg11_bn.eval()

# device
use_gpu_if_available = True
use_CUDA = use_gpu_if_available and torch.cuda.is_available()
if use_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('# using', device)

nn = vgg11_bn.to(device)


# setting the image transformation
#    cf. https://pytorch.org/docs/stable/torchvision/models.html
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.CenterCrop((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# dataset
dV = torchvision.datasets.ImageNet(p, split='val', transform=trans)
#print(dV)

# dataloader
bsize = 64
#dl = torch.utils.data.DataLoader(dV, batch_size=bsize, shuffle=True)
dl = torch.utils.data.DataLoader(dV, batch_size=bsize, shuffle=True, pin_memory=use_CUDA, num_workers=8)
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
