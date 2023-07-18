
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *

__all__ = ['R_Test','Q_Test']

class R_Test(nn.Module):
    def __init__(self, num_classes:int) -> None:
        super(R_Test, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3,padding=1)
        self.conv1 = nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,padding=1)
        
        self.fc0 = nn.Linear(28*28*10,128)
        self.fc1 = nn.Linear(128,num_classes)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x:Tensor) -> Tensor:
        x = self.conv0(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.relu(x)


        x = x.view(x.size(0), -1)

        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        
        return x


class Q_Test(nn.Module):
    def __init__(self, num_classes:int, w_bit:int = 1, a_bit:int = 2) -> None:
        super(Q_Test, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3,padding=1)
        self.conv1 = QuantizedConv2d(in_channels=10,out_channels=10,kernel_size=3,padding=1,k=w_bit)
                
        self.fc0 = QuantizedLinear(28*28*10,128,k=w_bit)
        self.fc1 = nn.Linear(128,num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.activation = quantize_activation(k=a_bit)


    def forward(self, x:Tensor) -> Tensor:
 
        x = self.conv0(x)
        x = self.relu(x)
        x = self.activation(x)
    
        x = self.conv1(x)
        x = self.relu(x)
        x = self.activation(x)


        x = x.view(x.size(0), -1)

        x = self.fc0(x)
        x = self.relu(x)
        x = self.activation(x)

        x = self.fc1(x)

        
        return x


