import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *



class ResBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, stride:int=1, w_bit:int=1, a_bit:int=2) -> None:
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.w_bit = w_bit
        self.a_bit = a_bit

        self.conv1 = QuantizedConv2d(
                                    in_channels=self.in_channel,
                                    out_channels=self.out_channel,
                                    kernel_size=3,
                                    stride=self.stride,
                                    padding=1,
                                    k=w_bit,
                                    )
        

        self.conv2 = QuantizedConv2d(
                                    in_channels=self.out_channel,
                                    out_channels=self.out_channel,
                                    kernel_size=3,
                                    stride=self.stride,
                                    padding=1,
                                    k=w_bit,
                                    )
        
        self.activation = quantize_activation(a_bit)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)


    def forward(self, x:Tensor) -> Tensor:
        return x