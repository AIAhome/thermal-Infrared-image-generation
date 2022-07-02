import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
# 定义参数
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=20, help="epoch to start training from")
parser.add_argument("--dataset_name", type=str, default="dataset", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
opt = parser.parse_args()
input_shape = (opt.channels, opt.img_height, opt.img_width)
os.makedirs("inference/%s"%opt.dataset_name, exist_ok=True)
# 初始化生成器和鉴别器
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
cuda = torch.cuda.is_available()
#cuda
if cuda:
    G_AB = G_AB.cuda()

if opt.epoch != 0:
    # 如果epoch不是从0 开始，则 Load 预训练模型
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # 否则初始化参数
    G_AB.apply(weights_init_normal)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("/data/shenliao/FLIR_ADAS_v2", transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1,
)
G_AB.eval()
for i, batch in enumerate(val_dataloader):
    real_A = Variable(batch["A"].type(Tensor))
    fake_B = G_AB(real_A)
    save_image(fake_B,"inference/%s/%d.png" % (opt.dataset_name,i), normalize=True)


    