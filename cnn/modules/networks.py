import torch
from torch import nn

import config

## CNN
class ResidualBlock(nn.Module):
    '''
    A residual block object that skips layers until stride > 1, i.e. the size of data shrinks
    '''
    
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        
        super(ResidualBlock,self).__init__()
        
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
            
    def forward(self,x):
        
        residual = x
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        x += self.shortcut(residual)
        
        x = nn.ReLU(True)(x)
        return x

    
class ForkCNN(nn.Module):
    '''
    CNN used for direct prediction of parameters.
    '''
    def __init__(self, batch_size, GPUs=1, 
                 nspec=config.data['nspec'], 
                 nfeatures=config.train['feature_number']):
        
        self.nfeatures = nfeatures
        self.batch = batch_size
        self.GPUs = GPUs
        
        super(ForkCNN, self).__init__()
        
        self.cnn_img = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, 2),
            
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128, 2),
            
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256, 2),
            
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
        )
        
        self.cnn_spec = nn.Sequential(
            
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 512, kernel_size=(nspec, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
        )
        
        ### Fully-connected layers
        self.fully_connected_layer = nn.Sequential(
            # make sure the first number is equal to the sum of final # of channels in both img and spec branches
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 32),
            nn.Linear(32, self.nfeatures),
            nn.Sigmoid()
        )

    
    def forward(self, x, y):
        
        x = self.cnn_img(x)
        
        y = self.cnn_spec(y)
        
        # Flatten
        x = x.view(int(self.batch),-1)
        y = y.view(int(self.batch),-1)
        
        # Concatenation
        z = torch.cat((x, y), -1)
        z = self.fully_connected_layer(z)
        
        return z

class DeconvNN(nn.Module):
    '''
    A deconv model in testing
    '''
    def __init__(self, batch_size, GPUs=1, 
                 nspec=config.data['nspec'], 
                 nfeatures=config.train['feature_number']):
        
        self.nfeatures = nfeatures
        self.batch = batch_size
        self.GPUs = GPUs
        
        super(DeconvNN, self).__init__()
        
        self.dnn_img = nn.Sequential(
            
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )
        
        ### Fully-connected layers
        self.linear = nn.Sequential(
            
            nn.Linear(self.nfeatures, 32),
            nn.Linear(32, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        )

    
    def forward(self, x):
        
        x = self.linear(x)
        
        x = x.view(int(self.batch),-1, 1, 1)
        
        x = self.dnn_img(x)
        
        return x

    
## NN calibration
class CaliNN(nn.Module):
    
    def __init__(self):
        super(CaliNN, self).__init__()
        
        self.main_net = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(4,5),
            #nn.ReLU(),
            nn.Linear(5,5),
            #nn.ReLU(),
            nn.Linear(5,5),
            #nn.ReLU(),
            nn.Linear(5,1),
        )

    
    def forward(self, x):
        
        x = self.main_net(x)
        return x
    
    


