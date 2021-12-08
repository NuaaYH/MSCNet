from MSCNet.utils import *
from torchvision.transforms import transforms
import os
from PIL import Image
from torch.utils import data
import numpy as np
import random

def dataset_info(dt):

    if dt == 'EORSSD':
        dt_mean = [0.3412, 0.3798, 0.3583]
        dt_std = [0.1148, 0.1042, 0.0990]
    return dt_mean, dt_std


def random_aug_transform():
    flip_h = transforms.RandomHorizontalFlip(p=1)
    flip_v = transforms.RandomVerticalFlip(p=1)
    angles = [0, 90, 180, 270]
    rot_angle = angles[np.random.choice(4)]
    rotate = transforms.RandomRotation((rot_angle, rot_angle))
    r = np.random.random()
    if r <= 0.25:
        flip_rot = transforms.Compose([flip_h, flip_v,rotate])
    elif r <= 0.5:
        flip_rot = transforms.Compose([flip_h,rotate])
    elif r <= 0.75:
        flip_rot = transforms.Compose([flip_v, flip_h,rotate])
    else:
        flip_rot = transforms.Compose([flip_v,rotate])
    return flip_rot



def RandomCrop(image, crop_shape,padding=10,mask=None,detail=None,):
    shape=image.size
    shape_pad=(shape[0]+2*padding,shape[1]+2*padding)
    image_pad=Image.new("RGB",(shape_pad[0],shape_pad[1]))
    mask_pad = Image.new("L", (shape_pad[0], shape_pad[1]))
    edge_pad = Image.new("L", (shape_pad[0], shape_pad[1]))
    image_pad.paste(image,(padding,padding))
    mask_pad.paste(mask, (padding, padding))
    edge_pad.paste(detail, (padding, padding))
    nh=random.randint(0,shape[0]-crop_shape[0])
    nw = random.randint(0, shape[1] - crop_shape[1])
    image_crop=image_pad.crop((nh,nw,nh+crop_shape[0],nw+crop_shape[1]))
    mask_crop = mask_pad.crop((nh, nw, nh + crop_shape[0], nw + crop_shape[1]))
    edge_crop = edge_pad.crop((nh, nw, nh + crop_shape[0], nw + crop_shape[1]))
    return image_crop,mask_crop,edge_crop

    
class EORSSD(data.Dataset):
    def __init__(self, root, mode, aug=False):
        self.aug = aug
        self.dt_mean, self.dt_std = dataset_info('EORSSD')
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        self.image_paths = [os.path.join(root, 'images', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, 'labels', prefix + '.png') for prefix in self.prefixes]
        self.edge_paths = [os.path.join(root, 'edges', prefix + '.png') for prefix in self.prefixes]
        self.image_transformation = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(self.dt_mean, self.dt_std)])
        self.label_transformation = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

    def __getitem__(self, index):
        if self.aug:
            flip_rot = random_aug_transform()
            image=Image.open(self.image_paths[index])
            label=Image.open(self.label_paths[index])
            edge=Image.open(self.edge_paths[index])
            image = self.image_transformation(flip_rot(image))
            label = self.label_transformation(flip_rot(label))
            edge = self.label_transformation(flip_rot(edge))
            return image, label, edge
        else:
            image = self.image_transformation(Image.open(self.image_paths[index]))
            label = self.label_transformation(Image.open(self.label_paths[index]))
            name = self.prefixes[index]
            return image, label,name
    def __len__(self):
        return len(self.prefixes)
    
    
