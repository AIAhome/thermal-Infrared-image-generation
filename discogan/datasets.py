import glob
import os
# from types import NoneType
import torch

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


# 构建Image数据集
class ImageDataset(Dataset):
    def __init__(self, A_path, B_path, A_transforms=None, B_transfroms=None):
        self.A_transform = transforms.Compose(A_transforms)
        self.B_transform = transforms.Compose(B_transfroms)  # transform

        self.A_files = sorted(glob.glob(A_path + '/*.*'))  # files list
        self.B_files = sorted(glob.glob(B_path + '/*.*'))

    def __getitem__(self, index):

        # 打开的图像是PIL格式的，取%只是为了放置溢出
        rgb_imgs = Image.open(self.A_files[index % len(self.A_files)])
        thermal_imgs = Image.open(self.B_files[index % len(self.B_files)])

        # 对读入图像做变换
        img_A = self.A_transform(rgb_imgs)
        img_B = self.B_transform(thermal_imgs)

        return {'A': img_A, 'B': img_B}

    def __len__(self, is_A=True):
        if is_A:
            return len(self.A_files)
        else:
            return len(self.B_files)
