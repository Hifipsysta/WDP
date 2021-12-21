


import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(64 * 4 * 4, 512) 
        self.linear2 = nn.Linear(512, 10) 
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4) 
        x = self.dropout(x)
        x = F.relu(self.linear1(x)) 
        x = self.dropout(x)
        x = self.linear2(x) 
        return x

class View(nn.Module):
    """
        Implements a reshaping module.
        Allows to reshape a tensor between NN layers.
    """

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)

class MNISTConvNet(nn.Module):

    def __init__(self, nChannels=1, ndf=64, filterSize=5, w_out=4, h_out=4, nClasses=10):
        super(MNISTConvNet, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nChannels, ndf, filterSize),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(ndf),
            nn.Conv2d(ndf, ndf, filterSize),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2),
            View(-1, ndf * w_out * h_out),
            #PrintLayer("View"),
            #View(-1, 784),
            nn.Linear(ndf * w_out * h_out, 384),
            nn.SELU(inplace=True),
            nn.Linear(384, 192),
            nn.SELU(inplace=True),
            nn.Linear(192, nClasses),
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class CIFARConvNet(nn.Module):
    
    def __init__(self, nChannels=3, ndf=64, filterSize=5, w_out=5, h_out=5, nClasses=10):
        super(CIFARConvNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, ndf, filterSize),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(ndf),
            nn.Conv2d(ndf, ndf, filterSize),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            View(-1, ndf * w_out * h_out),
            #PrintLayer("View"),
            nn.Linear(ndf * w_out * h_out, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, nClasses),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x
