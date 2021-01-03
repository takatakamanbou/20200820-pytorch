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
            W[i*8+j, ::] = basis[i, j, ::]

    return W


def kernelDCT2d2():

    W = np.zeros((192, 3, 8, 8))
    basis = DCT2d.DCT2d(8)
    c = 0
    for i in range(3):
        for j in range(8):
            for k in range(8):
                W[c, i, ::] = basis[j, k, ::]
                c+= 1

    return W



class DCTnet(nn.Module):

    def __init__(self):

        super(DCTnet, self).__init__()

        self.conv01 = nn.Conv2d(3, 64, 8, stride = 8, bias = False)
        W = kernelDCT2d()
        self.conv01.weight = nn.Parameter(torch.Tensor(W), requires_grad=False)
        # (28, 28, 64)

        self.conv02a = nn.Conv2d(64, 256, 3)
        self.conv02b = nn.Conv2d(256, 256, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        # (14, 14, 256)

        self.conv03a = nn.Conv2d(256, 512, 3)
        self.conv03b = nn.Conv2d(512, 512, 3)
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

        X = F.relu(self.conv02a(X))
        X = F.relu(self.conv02b(X))
        X = self.pool1(X)

        X = F.relu(self.conv03a(X))
        X = F.relu(self.conv03b(X))
        X = self.pool2(X)

        X = self.avepool1(X)

        X = X.view(-1, 512*7*7)

        X = F.relu(self.fc1(X))
        X = self.dropout1(X)
        X = F.relu(self.fc2(X))
        X = self.dropout2(X)
        X = self.fc3(X)

        return X


class DCTnet2(nn.Module):

    def __init__(self):

        super(DCTnet2, self).__init__()

        self.conv01 = nn.Conv2d(3, 192, 8, stride = 8, bias = False)
        W = kernelDCT2d2()
        self.conv01.weight = nn.Parameter(torch.Tensor(W), requires_grad=False)
        # (28, 28, 192)

        self.conv02a = nn.Conv2d(192, 256, 3)
        self.conv02b = nn.Conv2d(256, 256, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        # (14, 14, 256)

        self.conv03a = nn.Conv2d(256, 512, 3)
        self.conv03b = nn.Conv2d(512, 512, 3)
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

        X = F.relu(self.conv02a(X))
        X = F.relu(self.conv02b(X))
        X = self.pool1(X)

        X = F.relu(self.conv03a(X))
        X = F.relu(self.conv03b(X))
        X = self.pool2(X)

        X = self.avepool1(X)

        X = X.view(-1, 512*7*7)

        X = F.relu(self.fc1(X))
        X = self.dropout1(X)
        X = F.relu(self.fc2(X))
        X = self.dropout2(X)
        X = self.fc3(X)

        return X


class DCTnet2v2(nn.Module):

    def __init__(self):

        super(DCTnet2v2, self).__init__()

        self.conv01r = nn.Conv2d(3, 64, 8, stride = 8, bias = False)
        self.conv01g = nn.Conv2d(3, 64, 8, stride = 8, bias = False)
        self.conv01b = nn.Conv2d(3, 64, 8, stride = 8, bias = False)
        #W = kernelDCT2d()
        W = kernelDCT2d2()[:, 0, np.newaxis, :, :]
        self.conv01r.weight = nn.Parameter(torch.Tensor(np.copy(W)), requires_grad=False)
        self.conv01g.weight = nn.Parameter(torch.Tensor(np.copy(W)), requires_grad=False)
        self.conv01b.weight = nn.Parameter(torch.Tensor(np.copy(W)), requires_grad=False)
        # (224, 224, 3) => (224, 224, 1) x 3 => (28, 28, 64) x 3 => (28, 28, 192)

        self.conv02a = nn.Conv2d(192, 256, 3)
        self.conv02b = nn.Conv2d(256, 256, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        # (14, 14, 256)

        self.conv03a = nn.Conv2d(256, 512, 3)
        self.conv03b = nn.Conv2d(512, 512, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        # (7, 7, 512)

        self.avepool1 = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    
    def forward(self, X):

        Xr, Xg, Xb = torch.split(X, 1, dim=1)
        #print('### (0) XX.shape =', XX[0].shape)
        #print('### (1) X.shape =', X.shape, X.shape[2:])
        print('### Xr.shape =', Xr.shape)
        Xr = self.conv01r(Xr)
        Xg = self.conv01g(Xg)
        Xb = self.conv01b(Xb)
        X = torch.cat((Xr, Xg, Xb), dim=1)
        print('### (2) X.shape =', X.shape)

        X = F.relu(self.conv02a(X))
        X = F.relu(self.conv02b(X))
        X = self.pool1(X)

        X = F.relu(self.conv03a(X))
        X = F.relu(self.conv03b(X))
        X = self.pool2(X)

        X = self.avepool1(X)

        X = X.view(-1, 512*7*7)

        X = F.relu(self.fc1(X))
        X = self.dropout1(X)
        X = F.relu(self.fc2(X))
        X = self.dropout2(X)
        X = self.fc3(X)

        return X



class DCTnet3(nn.Module):

    def __init__(self):

        super(DCTnet3, self).__init__()

        self.conv01 = nn.Conv2d(3, 192, 8, stride = 4, bias = False)
        W = kernelDCT2d2()
        self.conv01.weight = nn.Parameter(torch.Tensor(W), requires_grad=False)
        # (28, 28, 192)

        self.conv02a = nn.Conv2d(192, 256, 3)
        self.conv02b = nn.Conv2d(256, 256, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        # (14, 14, 256)

        self.conv03a = nn.Conv2d(256, 512, 3)
        self.conv03b = nn.Conv2d(512, 512, 3)
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

        X = F.relu(self.conv02a(X))
        X = F.relu(self.conv02b(X))
        X = self.pool1(X)

        X = F.relu(self.conv03a(X))
        X = F.relu(self.conv03b(X))
        X = self.pool2(X)

        X = self.avepool1(X)

        X = X.view(-1, 512*7*7)

        X = F.relu(self.fc1(X))
        X = self.dropout1(X)
        X = F.relu(self.fc2(X))
        X = self.dropout2(X)
        X = self.fc3(X)

        return X


