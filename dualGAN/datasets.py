import glob
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, A_path, B_path, A_transforms=None, B_transforms = None):
        self.A_transform = transforms.Compose(A_transforms)# 定义变换方法
        self.B_transform = transforms.Compose(B_transforms)

        self.A_files = sorted(glob.glob(A_path + "/*.*")) # 读取文件数据
        self.B_files = sorted(glob.glob(B_path + "/*.*"))

    def __getitem__(self, index):

        rgb_imgs = Image.open(self.A_files[index % len(self.A_files)]) # 打开的图像是PIL格式的，取%只是为了放置溢出
        thermal_imgs = Image.open(self.B_files[index % len(self.B_files)])
        # w, h = img.size
        # img_A = img.crop((0, 0, w / 2, h)) # 示例是那个楼房方块到实景的迁移，每个图片是一个数据，左边一半是真实图片，右边一半是方块样的图像
        # img_B = img.crop((w / 2, 0, w, h))

        # img.convert("L") 可以把图像转变为灰度格式，通过img的mode属性可以看出其格式

        # if np.random.random() < 0.5: # 增加数据多样性？随机做上下翻转的操作
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB") # np.array(img)得到的是(w,h,c)格式
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.A_transform(rgb_imgs) # 进行图像变换
        img_B = self.B_transform(thermal_imgs)

        return {"A": img_A, "B": img_B}

    def __len__(self,is_A=True):
        if is_A:
            return len(self.A_files)
        else:
            return len(self.B_files)
