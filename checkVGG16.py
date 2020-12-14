import numpy as numpy
import torch
import torchvision
import datetime

# path to ILSVRC2012
p = '/mnt/data/ILSVRC2012'

# loading VGG16 pretrained model
#vgg16 = torchvision.models.vgg16(pretrained=True)
#vgg16.eval()

# device
use_gpu_if_available = False
use_CUDA = use_gpu_if_available and torch.cuda.is_available()
if use_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('# using', device)

#nn = vgg16.to(device)


# setting the image transformation
#    cf. https://pytorch.org/docs/stable/torchvision/models.html
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.CenterCrop((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# dataset
dL = torchvision.datasets.ImageNet(p, split='train', transform=trans)
#print(dV)

# dataloader
bsize = 100
dl = torch.utils.data.DataLoader(dL, batch_size=bsize, shuffle=False, pin_memory=use_CUDA, num_workers=8)
nbatch = len(dl)

s1 = datetime.datetime.now()

ib = 12609
print(dl[ib].shape)

for ib, rv in enumerate(dl):
    if ib == 12609:
        X, lab = rv[0].to(device), rv[1].to(device)
        print(f'# {ib}  {X.shape}')
        for i, label in enumerate(lab):
            print(i, lab)



s2 = datetime.datetime.now()

print(f'{s2 - s1}')
