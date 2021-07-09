""" Parts of the model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)



class Down3(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_3conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            TripleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_3conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels=1, upscale=1, method='bilinear'):
        super().__init__()
        
        self.conv =  OutConv(in_channels)#(128, 1, 1)
        #self.stride=stride
        # if bilinear, use the normal convolutions to reduce the number of channels
        if method=='bilinear':
            stride=2**upscale
            self.up = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
        else:
            stride=2**upscale
            self.up = nn.ConvTranspose2d(in_channels=1 , out_channels=out_channels, kernel_size=stride*2, stride=stride)
            # w.shape = (1,1,kH,kW)
            #output.shape = [1,1,oH,oW], where oH=(iH-1)*S+k (here k=kH=kW)
            # (iH/2-1)*2+4=iH+2

    def forward(self, x, crop_size):
            x = self.up(self.conv(x))
            return crop(x, crop_size[0], crop_size[1])



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels=1, uniform=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if uniform:
            self.conv.weight.data.fill_(1/in_channels)
            self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)



def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]