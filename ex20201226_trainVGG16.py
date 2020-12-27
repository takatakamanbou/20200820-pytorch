import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import datetime
import sys
import os

import ilsvrc2012


if __name__ == '__main__':

    if len(sys.argv) < 2 or 3 < len(sys.argv):
        print(f'usage: {sys.argv[0]} epoch_now [epoch_end]')
        exit()

    epoch_start = int(sys.argv[1])
    if len(sys.argv) == 3:
        epoch_end = int(sys.argv[2])
    else:
        epoch_end = epoch_start
    epoch_prev = epoch_start - 1
    assert epoch_prev >= 0


    # device
    use_gpu_if_available = True
    use_CUDA = use_gpu_if_available and torch.cuda.is_available()
    if use_CUDA:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('# using', device)

    # loading VGG11-bn pretrained model
    vgg16 = torchvision.models.vgg16(pretrained=False)
    print(vgg16)

    # loading the parameters
    fnParam_prev = f'data/ex20201226_trainVGG16_epoch{epoch_prev:03d}.pth'
    if epoch_prev != 0:
        with open(fnParam_prev, mode='rb') as f:
            vgg16.load_state_dict(torch.load(f))

    nn = vgg16.to(device)
    nn.train()

    # dataset & dataloader
    dL = ilsvrc2012.datasets('L')
    bsize = 256
    dl = torch.utils.data.DataLoader(dL, batch_size=bsize, shuffle=True, pin_memory=use_CUDA, num_workers=16)
    nbatch = len(dl)

    # optimizer
    #optimizer = optim.SGD(nn.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(nn.parameters(), lr=0.0001, weight_decay=0.0)

    nb = 1024 // bsize
    ncList = np.zeros(nb, dtype=int)
    lossList = np.zeros(nb)

    for epoch_now in range(epoch_start, epoch_end+1):

        print()
        print(f'##### epoch {epoch_now} #####')
        print()

        fnParam_now = f'data/ex20201226_trainVGG16_epoch{epoch_now:03d}.pth'
        if os.path.exists(fnParam_now):
            print(f'{fnParam_now} exists!')
            exit()

        s1 = datetime.datetime.now()

        for ib, rv in enumerate(dl):
            X, lab = rv[0].to(device), rv[1].to(device)
            optimizer.zero_grad()
            output = F.log_softmax(nn(X), dim=1)
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
                sys.stdout.flush()

        s2 = datetime.datetime.now()

        print(s2-s1)

        print(f'# writing to {fnParam_now}')
        #with open(fnParam_now, mode='wb') as f:
        #    torch.save(nn.state_dict(), f)    


