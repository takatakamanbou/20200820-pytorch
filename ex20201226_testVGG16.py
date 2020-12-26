import numpy as numpy
import torch
import torch.nn.functional as F
import torchvision
import datetime
import sys

import ilsvrc2012


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} epoch')
        exit()

    epoch = int(sys.argv[1])

    # loading VGG16 model
    vgg16 = torchvision.models.vgg16(pretrained=False)
    print(vgg16)

    # loading the parameters
    fnParam = f'data/ex20201226_trainVGG16_epoch{epoch:03d}.pth'
    with open(fnParam, mode='rb') as f:
        vgg16.load_state_dict(torch.load(f))

    # device
    use_gpu_if_available = True
    use_CUDA = use_gpu_if_available and torch.cuda.is_available()
    if use_CUDA:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('# using', device)

    nn = vgg16.to(device)
    nn.eval()

    # dataset & dataloader
    dT = ilsvrc2012.datasets('T')
    bsize = 64
    dl = torch.utils.data.DataLoader(dT, batch_size=bsize, shuffle=False, pin_memory=use_CUDA, num_workers=16)
    nbatch = len(dl)

    s1 = datetime.datetime.now()

    ncorrect = 0
    ntotal = 0
    with torch.no_grad():
        for ib, rv in enumerate(dl):
            X, lab = rv[0].to(device), rv[1].to(device)
            output = F.log_softmax(nn(X), dim=1)
            pred = output.max(1, keepdim=True)[1]
            nc = pred.eq(lab.view_as(pred)).sum().item()
            nd = len(X)
            sb = datetime.datetime.now()
            print(f'# {ib}  {nc}/{nd}')
            sys.stdout.flush()
            ncorrect += nc
            ntotal += nd

    s2 = datetime.datetime.now()

    nerror = ntotal - ncorrect
    #print(f'{nerror}/{ntotal} = {nerror/ntotal:.2f}')
    print(f'{ncorrect}/{ntotal} = {ncorrect/ntotal:.2f}')
    print(f'{s2 - s1}')
