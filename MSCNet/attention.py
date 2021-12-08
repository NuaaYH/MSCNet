from MSCNet.utils import *
from MSCNet.modules import *
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self,k=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k, padding=k//2,bias=False) # infer a one-channel attention map

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True) # [B, 1, H, W], average
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True) # [B, 1, H, W], max
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1) # [B, 2, H, W]
        att_map = F.sigmoid(self.conv(ftr_cat)) # [B, 1, H, W]
        return att_map


class ChannelAttention(nn.Module):
    def __init__(self, in_planes,outc,ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(nn.Linear(in_planes,in_planes//ratio),nn.ReLU(inplace=True),nn.Linear(in_planes//ratio,outc))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,_,_=x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(2).unsqueeze(3)



class PA(nn.Module):
    def __init__(self):
        super(PA, self).__init__()
        self.conv1=convbnrelu(128,64)
        self.conv2=convbnrelu(128,64)
        self.conv3=convbnrelu(128,64)
        self.conv4=convbnrelu(128,64)
        self.conv5=convbnrelu(128,64)
        self.conv6=convbnrelu(128,64)
        self.a1=ChannelAttention(64,64,4)
        self.a2=ChannelAttention(64,64,4)
        self.a3=ChannelAttention(64,64,4)
        self.s1=SpatialAttention(k=3)
        self.s2=SpatialAttention(k=3)

    def forward(self, x1,x2,x3,x4):

        p1=torch.cat([x1,US2(x2)],dim=1)
        p1=self.conv1(p1)
        p1=self.s1(p1)*p1

        p2 = torch.cat([x2, US2(x3)], dim=1)
        p2 = self.conv2(p2)

        p3 = torch.cat([x3, US2(x4)], dim=1)
        p3 = self.conv3(p3)
        p3=self.a1(p3)*p3

        s1=torch.cat([p1,US2(p2)],dim=1)
        s1=self.conv4(s1)
        s1=self.s2(s1)*s1

        s2 = torch.cat([p2, US2(p3)], dim=1)
        s2 = self.conv5(s2)
        s2=self.a2(s2)*s2

        out=torch.cat([s1,US2(s2)],dim=1)
        out=self.conv6(out)
        out=self.a3(out)*out

        return out



