import numpy as numpy
import torch
import torchvision
import datetime

p = '/mnt/data/ILSVRC2012'


vgg16 = torchvision.models.vgg16(pretrained=true)


# この transform は，動作確認のための最小限のもの．
# 実際の実験の際は，画素値の平均を引く操作等，pre-trained モデルで使われたのを再現しないといけない
trans = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((224, 224)),
    torchvision.transforms.ToTensor(),
])




s0 = datetime.datetime.now()

#dL = torchvision.datasets.ImageNet(p)
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

for ib, rv in enumerate(dl):
    if ib == N:
        break
    X, lab = rv
    sb = datetime.datetime.now()
    print('#', ib, X.shape, lab)

s4 = datetime.datetime.now()
print(f'# loading {N} batches (batchsize = {bsize}):', s4 - s3)
