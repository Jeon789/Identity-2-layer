import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F  
    
    
    
class Model(nn.Module):    
    def __init__(self):
        super(Model, self).__init__()
        
        
        self.conv1a = nn.Conv2d(3,64,3,padding=1)
        self.BN1a = nn.BatchNorm2d(64)  # + LR
        
        #1st Block
        self.conv1b = nn.Conv2d(64,64,3,padding=1)
        self.BN1b = nn.BatchNorm2d(64)  # + LR
        self.conv1c = nn.Conv2d(64,64,3,padding=1)
        self.BN1c = nn.BatchNorm2d(64)  # + Adding + LR + AvgPool
        
        
        
        #Input = 16 x 16 x 64
        self.conv2a = nn.Conv2d(64,128,3,padding=1)
        self.BN2a = nn.BatchNorm2d(128)  # + LR
        
        #2st Block
        self.conv2b = nn.Conv2d(128,128,3,padding=1)
        self.BN2b = nn.BatchNorm2d(128)  # + LR
        self.conv2c = nn.Conv2d(128,128,3,padding=1)
        self.BN2c = nn.BatchNorm2d(128)  # + Adding + LR + AvgPool
        
        
        
        #Input = 8 x 8 x 128
        self.conv3a = nn.Conv2d(128,256,3,padding=1)
        self.BN3a = nn.BatchNorm2d(256)  # + LR
        
        #3st Block
        self.conv3b = nn.Conv2d(256,256,3,padding=1)
        self.BN3b = nn.BatchNorm2d(256)  # + LR
        self.conv3c = nn.Conv2d(256,256,3,padding=1)
        self.BN3c = nn.BatchNorm2d(256)  # + Adding + LR + AvgPool
        
        
        #Input = 4 x 4 x 256
        self.conv3a = nn.Conv2d(256,512,3,padding=1)
        self.BN3a = nn.BatchNorm2d(512)  # + LR
        
        #4st Block
        self.conv3b = nn.Conv2d(512,512,3,padding=1)
        self.BN3b = nn.BatchNorm2d(512)  # + LR
        self.conv3c = nn.Conv2d(512,512,3,padding=1)
        self.BN3c = nn.BatchNorm2d(512)  # + Adding + LR + AvgPool
        
        
        
        #Input = 2 x 2 x 512
        self.conv3a = nn.Conv2d(512,1024,3,padding=1)
        self.BN3a = nn.BatchNorm2d(1024)  # + LR
        
        #5st Block
        self.conv3b = nn.Conv2d(1024,1024,3,padding=1)
        self.BN3b = nn.BatchNorm2d(1024)  # + LR
        self.conv3c = nn.Conv2d(1024,1024,3,padding=1)
        self.BN3c = nn.BatchNorm2d(1024)  # + Adding + LR + AvgPool
        #Now 1 x 1 x 1024
        

        self.fc1 = nn.Linear(1024,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)


    def forward(self, x):

        #input = 32 x 32 x 3
        x1a = nn.LeakyReLU(0.2)(self.BN1a(self.conv1a(x)))
        
        x1b = nn.LeakyReLU(0.2)(self.BN1b(self.conv1b(x1a)))
        x1c = self.BN1c(self.conv1c(x1b))
        x1c = nn.LeakyReLU(0.2)(x1c + x1a)
        x1 = nn.MaxPool2d(2,2)(x1c)
        
        #input  = 16 x 16 x 64
        x2a = nn.LeakyReLU(0.2)(self.BN2a(self.conv2a(x1)))
        
        x2b = nn.LeakyReLU(0.2)(self.BN2b(self.conv2b(x2a)))
        x2c = self.BN2c(self.conv2c(x2b))
        x2c = nn.LeakyReLU(0.2)(x2c + x2a)
        x2 = nn.MaxPool2d(2,2)(x2c)
        
        #input  = 8 x 8 x 128
        x3a = nn.LeakyReLU(0.2)(self.BN3a(self.conv3a(x2)))
        
        x3b = nn.LeakyReLU(0.2)(self.BN3b(self.conv3b(x3a)))
        x3c = self.BN3c(self.conv3c(x3b))
        x3c = nn.LeakyReLU(0.2)(x3c + x3a)
        x3 = nn.MaxPool2d(2,2)(x3c)
        
        #input  = 4 x 4 x 256
        x4a = nn.LeakyReLU(0.2)(self.BN4a(self.conv4a(x3)))
        
        x4b = nn.LeakyReLU(0.2)(self.BN4b(self.conv4b(x4a)))
        x4c = self.BN4c(self.conv4c(x4b))
        x4c = nn.LeakyReLU(0.2)(x4c + x4a)
        x4 = nn.MaxPool2d(2,2)(x4c)
        
        #input  = 2 x 2 x 512
        x5a = nn.LeakyReLU(0.2)(self.BN5a(self.conv5a(x4)))
        
        x5b = nn.LeakyReLU(0.2)(self.BN5b(self.conv5b(x5a)))
        x5c = self.BN5c(self.conv5c(x5b))
        x5c = nn.LeakyReLU(0.2)(x5c + x5a)
        x5 = nn.MaxPool2d(2,2)(x5c)
        #output = 1 x 1 x1024
        
        
        x5 = torch.squeeze(x5)
        output = self.fc1(self.fc2(self.fc3(x5)))
        
        #make a dict to get feature maps at every step
        dict = {}
        dict['x1a'] = x1a
        dict['x1b'] = x1b
        dict['x1c'] = x1c
        dict['x2a'] = x2a
        dict['x2b'] = x2b
        dict['x3c'] = x2c
        dict['x3a'] = x3a
        dict['x3b'] = x3b
        dict['x3c'] = x3c
        dict['x4a'] = x4a
        dict['x4b'] = x4b
        dict['x4c'] = x4c
        dict['x5a'] = x5a
        dict['x5b'] = x5b
        dict['x5c'] = x5c
        dict['output'] = output
        
        #x = self.softmax(x)
        return dict
    
model = Model()
print(model)
