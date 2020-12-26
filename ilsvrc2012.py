import torchvision
import torchvision.transforms as T

# path to ILSVRC2012
path = '/mnt/data/ILSVRC2012'

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transL = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalize
])

# cf. https://pytorch.org/docs/stable/torchvision/models.html
transT = torchvision.transforms.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize
])


def datasets(LT):

    assert LT == 'L' or LT == 'T'

    if LT == 'L':
        ds = torchvision.datasets.ImageNet(path, split='train', transform=transL)
    else:
        ds = torchvision.datasets.ImageNet(path, split='val', transform=transT)

    return ds
