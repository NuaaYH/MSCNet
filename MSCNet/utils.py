from MSCNet.packages import *


import torch
import numpy as np
import torch.nn.functional as F


def align_number(number, N):
    assert type(number) == int
    num_str = str(number)
    assert len(num_str) <= N
    return (N - len(num_str)) * '0' + num_str


def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y


def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed


def convert2img(x):
    return Image.fromarray(x*255).convert('L')


def save_smap(smap, path):
    smap = convert2img(min_max_normalization(unload(smap)))
    smap.save(path)
    
    
def cache_model(model, path):
    torch.save(model.state_dict(), path)
        
        
def initiate(md, path):
    md.load_state_dict(torch.load(path))

def initiate0(net, path):
    mbv = models.mobilenet_v2(pretrained=True)
    new_state_dict = mbv.state_dict()
    dd = net.state_dict()

    dd = {k: v for k, v in dd.items() if k in new_state_dict}
    net.load_state_dict(dd)

        
def DS2(x):
    return F.avg_pool2d(x, 2)


def DS4(x):
    return F.avg_pool2d(x, 4)


def DS8(x):
    return F.avg_pool2d(x, 8)


def DS16(x):
    return F.avg_pool2d(x, 16)


def US2(x):
    """if size!=None:
        return F.interpolate(x, size=size, mode='bilinear')
    else:"""
    return F.interpolate(x, scale_factor=2, mode='bilinear')


def US4(x):
    return F.interpolate(x, scale_factor=4, mode='bilinear')


def US8(x):
    return F.interpolate(x, scale_factor=8, mode='bilinear')


def US16(x):
    return F.interpolate(x, scale_factor=16, mode='bilinear')


def RC(F, A):
    return F * A + F


def clip(inputs,rho=1e-15,mu=1-1e-15):
    return inputs*(mu-rho)+rho


def BCELoss_OHEM(batch_size, pred, gt, num_keep):
    loss = torch.zeros(batch_size).cuda()
    for b in range(batch_size):
        loss[b] = F.binary_cross_entropy(pred[b,:,:,:], gt[b,:,:,:])
        sorted_loss, idx = torch.sort(loss, descending=True)
        keep_idx = idx[0:num_keep]  
        ohem_loss = loss[keep_idx]  
        ohem_loss = ohem_loss.sum() / num_keep
    return ohem_loss


def proc_loss(losses, num_total, prec=4):
    loss_for_print = []
    for l in losses:
        loss_for_print.append(np.around(l / num_total, prec))
    return loss_for_print

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()


