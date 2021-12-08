from torch import nn
from torch import Tensor
from MSCNet.modules import *
from MSCNet.attention import *
import torch
import os
import math

"""
IMPORTANT:
To adapt it to the SOD task, weremove the global average pooling layer and the last fully-connected layer from the backbon
"""

__all__ = ["MobileNetV2"]

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.d1=MSCE(64,80)
        self.d2 = MSCE(64, 88)
        self.d3 = MSCE(64, 96)
        self.d4 = MSCE(64, 416)

        self.agg=PA()

    def forward(self, F1,F2,F3,F4,F5):

        P4 = torch.cat([F4, US2(F5)], dim=1)
        P4 = self.d4(P4)
        P3 = torch.cat([F3, US2(P4)], dim=1)
        P3 = self.d3(P3)
        P2 = torch.cat([F2, US2(P3)], dim=1)
        P2 = self.d2(P2)
        P1 = torch.cat([F1, US2(P2)], dim=1)
        P1 = self.d1(P1)

        S=self.agg(P1,P2,P3,P4)
        return S

mob_conv1_2 = mob_conv2_2 = mob_conv3_3 = mob_conv4_3 = mob_conv5_3 = None

def conv_1_2_hook(module, input, output):
    global mob_conv1_2
    mob_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global mob_conv2_2
    mob_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global mob_conv3_3
    mob_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global mob_conv4_3
    mob_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global mob_conv5_3
    mob_conv5_3 = output
    return None


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.mbv = models.mobilenet_v2(pretrained=True).features

        self.mbv[1].register_forward_hook(conv_1_2_hook)
        self.mbv[3].register_forward_hook(conv_2_2_hook)
        self.mbv[6].register_forward_hook(conv_3_3_hook)
        self.mbv[13].register_forward_hook(conv_4_3_hook)
        self.mbv[17].register_forward_hook(conv_5_3_hook)

    def forward(self, x: Tensor) -> Tensor:
        global mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3
        self.mbv(x)

        return mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3


class MobileNetV2(nn.Module):
    def __init__(self):

        super(MobileNetV2, self).__init__()
        #self.mbv = MobileNet()

        self.encoder=MobileNet()
        self.decoder = Decoder()
        self.head = nn.ModuleList([])
        for i in range(1):
            self.head.append(SalHead(64,3))

    def forward(self, x):
        #f1,f2,f3,f4,f5 = self.mbv(x)

        f1, f2, f3, f4, f5 = self.encoder(x)
        S= self.decoder(f1,f2,f3,f4,f5)
        sm = self.head[0](US2(S))

        return sm


