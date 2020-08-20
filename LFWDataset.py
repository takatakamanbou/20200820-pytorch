import numpy as np
import cv2
import torch
import os
import pickle


class LFWDataset(torch.utils.data.Dataset):

    ndatL = 4000
    ndatT = 1721
    imageShape = (128, 96, 3)
    ndim = np.prod(imageShape)


    def __init__(self, dataRoot='./data/lfw-selected/', LT='L'):

        self.dataRoot = os.path.normpath(dataRoot)
        assert os.path.isdir(self.dataRoot)
        print(f'# dataRoot = {self.dataRoot}')

        # reading the mean
        fn = os.path.join(self.dataRoot, 'meanL.npy')
        self.meanL = np.load(fn)

        assert LT == 'L' or LT == 'T'
        self.LT = LT
        dataLT = os.path.join(self.dataRoot, LT)
        if LT == 'L':
            self.ndat = LFWDataset.ndatL
        else:
            self.ndat = LFWDataset.ndatT

        # reading the images
        fn = os.path.join(dataLT, 'img0000.png')
        img = cv2.imread(fn)
        assert img.shape == LFWDataset.imageShape
        X = np.empty((self.ndat,) + img.shape)
        for i in range(self.ndat):
            fn = os.path.join(dataLT, f'img{i:04d}.png')
            #print(fn)
            X[i, ::] = cv2.imread(fn)

        # mean subtraction & scaling
        X -= self.meanL
        X /= 255.0

        self.X = X


    def __len__(self):

        return self.ndat

    
    def __getitem__(self, idx):

        return self.X[i, ::]
    


if __name__ == '__main__':

    #import torchvision

    ds = LFWDataset(LT='T')
    print(len(ds))
    #print(ds[0])
    print(ds.ndim)

    #dl = torch.utils.data.DataLoader()
