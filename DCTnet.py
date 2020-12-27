import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import DCT2d

def kernelDCT2d():

    W = np.empty((64, 3, 8, 8))
    basis = DCT2d.DCT2d(8)
    for i in range(8):
        for j in range(8):
            W[i*8+j, 0, ::] = basis[i, j, ::]
            W[i*8+j, 1, ::] = basis[i, j, ::]
            W[i*8+j, 2, ::] = basis[i, j, ::]

    return W


class DCTnet(nn.Module):

    def __init__(self):

        super(DCTnet, self).__init__()

        self.conv01 = nn.Conv2d(3, 64, 8, stride = 8, bias = False)
        W = kernelDCT2d()
        self.conv01.weight = nn.Parameter(torch.Tensor(W), requires_grad=False)
        # (28, 28, 64)

        self.conv02 = nn.Conv2d(64, 256, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        # (14, 14, 256)

        self.conv03 = nn.Conv2d(256, 512, 3)
        self.conv04 = nn.Conv2d(512, 512, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        # (7, 7, 512)

        self.avepool1 = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    
    def forward(self, X):

        X = self.conv01(X)

        X = F.relu(self.conv02(X))
        X = self.pool1(X)

        X = F.relu(self.conv03(X))
        X = F.relu(self.conv04(X))
        X = self.pool2(X)

        X = self.avepool1(X)

        X = X.view(-1, 512*7*7)

        X = F.relu(self.fc1(X))
        X = self.dropout1(X)
        X = F.relu(self.fc2(X))
        X = self.dropout2(X)
        X = self.fc3(X)

        return X

