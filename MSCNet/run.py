from MSCNet.dataset import *
from MSCNet.mobile import *
from MSCNet.modules import *
import os
import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
gpu_devices = list(np.arange(torch.cuda.device_count()))

output_folder = r'./Outputs/pred/MSCNet/EORSSD/Test'
ckpt_folder = r'./Checkpoints'
dataset_root = r'../Dataset/EORSSD'

batch_size = 6

def iou(pred, mask):
    inter = (pred * mask) .sum(dim=(2, 3))
    union = (pred + mask) .sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()

class BCEloss(nn.Module):
    def __init__(self):
        super(BCEloss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, sm,label):
        mask_loss = self.bce(sm,label) + 0.6*iou(sm,label)
        total_loss = mask_loss
        return [total_loss, mask_loss, mask_loss]

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

class Run:
    def __init__(self):
        self.train_set = EORSSD(dataset_root, 'train', aug=True)
        self.train_loader = data.DataLoader(self.train_set, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)  #true

        self.test_set = EORSSD(dataset_root, 'test', aug=False)
        self.test_loader = data.DataLoader(self.test_set, shuffle=False, batch_size=1, num_workers=4, drop_last=False)

        self.init_lr = 1e-4
        self.min_lr = 1e-7
        self.train_epoch = 40

        self.net = MobileNetV2()
        self.net.load_state_dict(torch.load(os.path.join(ckpt_folder, 'trained', 'best.pth')))
        self.loss=BCEloss()

    def train(self):
        self.net.train().cuda()
        max_F=0.86
        base, head = [], []
        for name, param in self.net.named_parameters():
            if 'mbv' in name:
                base.append(param)
            else:
                head.append(param)
        optimizer = optim.Adam([{'params': base}, {'params': head}], lr=self.init_lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_epoch,eta_min=self.min_lr)
        for epc in range(1, self.train_epoch + 1):
            records = [0] * 3
            N = 0
            optimizer.param_groups[0]['lr'] = 0.5 * optimizer.param_groups[1]['lr']  # for backbone
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr']
            for image, label, edge in tqdm(self.train_loader):
                # prepare input data\n",
                image, label, edge = image.cuda(), label.cuda(), edge.cuda()
                B = image.size(0)
                # forward\n",
                optimizer.zero_grad()
                sm = self.net(image)
                losses_list = self.loss(sm,label)
                # compute loss\n",
                total_loss = losses_list[0].mean()
                # record loss\n",
                N += B
                for i in range(len(records)):
                    records[i] += losses_list[i].mean().item() * B
                # backward\n",
                total_loss.backward()
                optimizer.step()
            # update learning rate\n",
            scheduler.step()
            F=self.test(epc)
            if F>max_F:
                cache_model(self.net, os.path.join(ckpt_folder, 'trained', 'trained.pth'))
                max_F=F
            if epc==self.train_epoch:
                cache_model(self.net, os.path.join(ckpt_folder, 'trained', 'last.pth'))
            # print training information\n",
            records = proc_loss(records, N, 4)
            print('epoch: {} || total loss: {} || mask loss: {} || edge loss: {}'
                  .format(epc, records[0], records[1], records[2]))
        print('finish training.'+'maxF:',max_F)

    def test(self,ep):
        self.net.eval().cuda()
        #print("params:",(count_param(self.net)/1e6))
        num_test = 0
        mae = 0.0
        F_value=0.0

        for image, label, prefix in self.test_loader:
            num_test += 1
            with torch.no_grad():
                image, label = image.cuda(), label.cuda()
                B=image.size(0)

                smap= self.net(image)

                mae += Eval_mae(smap, label)
                F_value += Eval_fmeasure(smap, label)
                if ep%4==0:
                    for b in range(B):
                        path = os.path.join(output_folder, prefix[b] + '.png')
                        save_smap(smap[b, ...], path)
        maxF=(F_value/num_test).max().item()
        meanF = (F_value / num_test).mean().item()
        mae=(mae/num_test)
        print('finish testing.', 'F—value : {:.4f}\t'.format(maxF),'mF—value : {:.4f}\t'.format(meanF),'MAE : {:.4f}\t'.format(mae))
        return maxF

def Eval_mae(pred,gt):
    pred=pred.cuda()
    gt=gt.cuda()
    with torch.no_grad():
        mae = torch.abs(pred - gt).mean()
        if mae == mae:  # for Nan
            return mae.item()

def Eval_fmeasure(pred,gt):
    beta2 = 0.3
    pred=pred.cuda()
    gt=gt.cuda()
    with torch.no_grad():
        pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                           torch.min(pred) + 1e-20)
        prec, recall = _eval_pr(pred, gt, 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
    return f_score#.max().item()


def _eval_pr(y_pred, y, num):
    if y_pred.sum() == 0: # a negative sample
        y_pred = 1 - y_pred
        y = 1 - y

    prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
    thlist = torch.linspace(0, 1 - 1e-10, num).cuda()

    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall


if __name__=='__main__':
    run=Run()
    #run.train()
    run.test(8)
