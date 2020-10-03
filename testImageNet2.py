import numpy as numpy
import torch
import torchvision

p = '/Volumes/share/data/ILSVRC2012'
#p = '/mnt/tlab-nas/share/data/ILSVRC2012'
#p = '/Users/takataka/Desktop/ILSVRC2012'

# この transform は，動作確認のための最小限のもの．
# 実際の実験の際は，画素値の平均を引く操作等，pre-trained モデルで使われたのを再現しないといけない
trans = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((224, 224)),
    torchvision.transforms.ToTensor(),
])


#dL = torchvision.datasets.ImageNet(p)
dV = torchvision.datasets.ImageNet(p, split='val', transform=trans)
print(dV)

dl = torch.utils.data.DataLoader(dV, batch_size=10, shuffle=True)


X, label = next(iter(dl))

