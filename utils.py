import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

__all__ = ['QuantizedConv2d','QuantizedLinear','QuantizedActivations']

def _quantize(k:int) -> Tensor:
    class quantize(torch.autograd.Function):

        @staticmethod
        def forward(ctx, r_i:Tensor)->Tensor:
            
            ctx.save_for_backward(r_i)

            if k==1: r_o = torch.sign(r_i)

            elif k==32: r_o = r_i

            else:
                n = 2**k - 1
                r_o = torch.round(n * r_i) / n

            return r_o

        @staticmethod
        def backward(ctx, g_o:Tensor) -> Tensor:
            return g_o.clone()
    
    return quantize().apply




class _quantize_weight(nn.Module):

    def __init__(self, k:int) -> None:
        super(_quantize_weight, self).__init__()
        self.k = k
        self.quantize = _quantize(k)


    def forward(self, r_i:Tensor) -> Tensor:
        '''
        quantize weight to k-bits
        '''

        if self.k==1:
            E = torch.mean(torch.abs(r_i)).detach()
            r_o = E * self.quantize(r_i / E)
                    
        else:
            tanh = torch.tanh(r_i)
            max = torch.max(torch.abs(tanh)).detach()
            clip = tanh / (2*max) + 0.5
            r_o = 2 * self.quantize(clip) - 1

        return r_o




class QuantizedActivations(nn.Module):

    def __init__(self, k:int=2) -> None:
        super(QuantizedActivations,self).__init__()
        
        self.k = k
        self.quantize = _quantize(k)


    def forward(self, r_i:Tensor) -> Tensor:
        '''
        quantize activations to k-bits
        '''
        if self.k==32:
            r_o = r_i
        else:
            r_o = self.quantize(torch.clamp(r_i,0,1))
        
        return r_o




class QuantizedConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 k: int = 1) -> None:
        
        super(QuantizedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.k = k
        self.q_weight = _quantize_weight(self.k)


    def forward(self, x:Tensor) -> Tensor:
        
        quantized_weight = self.q_weight(self.weight)

        return F.conv2d(input=x,
                        weight=quantized_weight,
                        bias=self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class QuantizedLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 k:int = 1) -> None:
        
        super(QuantizedLinear, self).__init__(in_features, out_features, bias, device, dtype)

        self.k = k
        self.q_weight = _quantize_weight(self.k)
        


    def forward(self, x:Tensor) -> Tensor:
        quantized_weight = self.q_weight(self.weight)

        return F.linear(
            input=x,
            weight=quantized_weight,
            bias=self.bias,
        )
    