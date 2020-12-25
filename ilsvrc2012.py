import torchvision
import torchvision.transforms as T

# path to ILSVRC2012
path = '/mnt/data/ILSVRC2012'

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.tor224, 0.225])

##### L #####

transL = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize
])

datasetsL = torchvision.datasets.ImageNet(path, split='train', transform=transL)

##### T #####

# cf. https://pytorch.org/docs/stable/torchvision/models.html
transT = torchvision.transforms.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize
])

datasetsT = torchvision.datasets.ImageNet(path, split='val', transform=transT)