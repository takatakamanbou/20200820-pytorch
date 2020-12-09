import numpy as numpy
import torch
import torchvision
import datetime

# path to ILSVRC2012
p = '/mnt/data/ILSVRC2012'

# loading VGG16 pretrained model
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.eval()

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
print(dV)

# dataloader
bsize = 10
dl = torch.utils.data.DataLoader(dV, batch_size=bsize, shuffle=True)
nbatch = len(dl)


N = 10  # number of batch to be fed

ncorrect = 0
ntotal = 0
with torch.no_grad():
    for ib, rv in enumerate(dl):
        if ib == N:
            break
        X, lab = rv
        output = vgg16(X)
        pred = output.max(1, keepdim=True)[1]
        nc = pred.eq(lab.view_as(pred)).sum().item()
        nd = len(X)
        sb = datetime.datetime.now()
        print(f'# {ib}  {nc}/{nd}')
        ncorrect += nc
        ntotal += nd

print(f'{nc}/{nd} = {nc/nd}'')

s4 = datetime.datetime.now()
