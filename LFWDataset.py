import numpy as np
import cv2
import torch
import os
import pickle


class LFWDataset(torch.utils.data.Dataset):

    ndatL = 4000
    ndatT = 1721
    imageShape = (96, 128, 3)
    ndim = np.prod(imageShape)


    def __init__(self, pathStr='./data/lfw-selected/', LT='L'):

        assert LT == 'L' or LT == 'T'
        self.LT = LT

        self.path = os.path.join(os.path.normpath(pathStr), LT)
        assert os.path.isdir(self.path)
        print(f'# {self.path}')

        # reading the attribute information
        with open(os.path.join(self.path, 'attributes.pickle'), 'rb') as f:
            rv = pickle.load(f)
        self.attrList = rv['list']
        self.ndat = len(self.attrList)

        # reading the images
        fn = os.path.join(self.path, 'img0000.png')
        img = cv2.imread(fn)
        X = np.empty((self.ndat,) + img.shape)
        for i in range(self.ndat):
            print(i)
            fn = os.path.join(self.path, f'img{i:04d}.png')
            X[i, ::] = cv2.imread(fn)
        self.X = X


    def __len__(self):

        return self.ndat

    
    def __getitem__(self, idx):

        return self.X[i, ::]
    


if __name__ == '__main__':

    ds = LFWDataset(LT='T')
    print(len(ds))
    #print(ds[0])
    print(ds.ndim)