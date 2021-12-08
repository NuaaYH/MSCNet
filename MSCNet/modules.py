from MSCNet.utils import *
import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks, use_bn, nl):
        # ic: input channels
        # oc: output channels
        # ks: kernel size
        # use_bn: True or False
        # nl: type of non-linearity, 'Non' or 'ReLU' or 'Sigmoid'
        super(ConvBlock, self).__init__()
        assert ks in [1, 3, 5, 7]
        assert isinstance(use_bn, bool)
        assert nl in ['None', 'ReLU', 'Sigmoid']
        self.use_bn = use_bn
        self.nl = nl
        if ks == 1:
            self.conv = nn.Conv2d(ic, oc, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, padding=(ks-1)//2, bias=False)
        if self.use_bn == True:
            self.bn = nn.BatchNorm2d(oc)
        if self.nl == 'ReLU':
            self.ac = nn.ReLU(inplace=True)
        if self.nl == 'Sigmoid':
            self.ac = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        if self.use_bn == True:
            y = self.bn(y)
        if self.nl != 'None':
            y = self.ac(y)
        return y


class SalHead(nn.Module):
    def __init__(self, in_channels, inter_ks):
        super(SalHead, self).__init__()
        self.conv_1 = ConvBlock(in_channels, in_channels//2, inter_ks, True, 'ReLU')
        self.conv_2 = ConvBlock(in_channels//2, in_channels//2, 3, True, 'ReLU')
        self.conv_3 = ConvBlock(in_channels//2, in_channels//8, 3, True, 'ReLU')
        self.conv_4 = ConvBlock(in_channels//8, 1, 1, False, 'Sigmoid')

    def forward(self, dec_ftr):
        dec_ftr_ups = dec_ftr
        outputs = self.conv_4(self.conv_3(self.conv_2(self.conv_1(dec_ftr_ups))))
        return outputs



class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class MSCE(nn.Module):
    def __init__(self, inc1, inc2, size=0):
        super(MSCE, self).__init__()
        self.s = size
        self. inc1=inc1
        self. inc2=inc2
        self.conv1=convbnrelu(inc2,inc1)
        self.conv2=convbnrelu(inc2,inc1)
        self.c0=nn.Sequential(
            nn.Conv2d(inc1, inc1, 3, padding=1, bias=False),
            nn.BatchNorm2d(inc1),
            nn.ReLU(inplace=True)

        )
        self.c1 = nn.Sequential(
            nn.Conv2d(inc1, inc1, (1, 3), padding=0, bias=False),
            nn.Conv2d(inc1, inc1, (3,1), padding=1,bias=False),
            nn.BatchNorm2d(inc1),
            nn.ReLU(inplace=True)
        )
        self.c11 = nn.Sequential(
            nn.Conv2d(inc1, inc1, (1, 5), padding=0, bias=False),
            nn.Conv2d(inc1, inc1, (5,1), padding=2,bias=False),
            nn.BatchNorm2d(inc1),
            nn.ReLU(inplace=True)
        )
        self.c12 = nn.Sequential(
            nn.Conv2d(inc1, inc1, (1, 7), padding=0, bias=False),
            nn.Conv2d(inc1, inc1, (7, 1), padding=3, bias=False),
            nn.BatchNorm2d(inc1),
            nn.ReLU(inplace=True)
        )

        self.conv=nn.Conv2d(inc2,inc1,kernel_size=1)
        self.conv0=nn.Sequential(convbnrelu(2*inc1,inc1),convbnrelu(inc1,inc1,k=3,p=1))
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.MaxPool2d(1)

    def forward(self, x1):
        n_b, c, _, _ = x1.shape
        if self.s == 1:
            x1 = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=True)
        va = self.avg(x1)
        vm=self.maxpool(x1)
        att = torch.sigmoid(self.conv(va)+self.conv(vm))
        f1=self.conv1(x1)
        f2=self.conv2(x1)
        out1=self.c0(f1)
        out2=self.c1(f2)+self.c11(f2)+self.c12(f2)
        out=torch.cat([out1,out2],dim=1)
        out=self.conv0(out)
        x1=out*att
        return x1
























