import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image


from tqdm import tqdm

from datasets import ImageDataset
from models import *

import torch
import matplotlib.pyplot as plt
import numpy as np

# timestamp = '20220609_100126' # 改进模型
timestamp = '20220608_220334' # 最初模型
data_path = './data'
output_path = './output'


in_channels = 3
out_channels = 3
n_cpu = 8
img_height = 256

############################################## keep the same as discogan.py ################################################
G_AB = GeneratorUNet(in_channels, out_channels)
G_AB.load_state_dict(torch.load('%s/saved_models/%s/G_AB_150.pth'%(output_path, timestamp)))


# Configure dataloader
A_transforms = [
    # transforms.Grayscale(),
    transforms.CenterCrop(size=1024),
    transforms.Resize(size=img_height, interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
]
B_transforms = [
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(size=img_height),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
]

dataloader = DataLoader(
    ImageDataset("%s/images_rgb_train/data" % data_path,
                 "%s/images_thermal_train/data" % data_path,
                 A_transforms=A_transforms,
                 B_transfroms=B_transforms),
    batch_size=1,
    shuffle=True,
    num_workers=n_cpu,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

############################################################################################################################

def inverse_normalize(tensor, mean, std):# 避免normalize之后显示出来的图像偏黑
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

G_AB.to(device)
G_AB.eval()

for i,data in tqdm(enumerate(dataloader),total=len(dataloader)):
    rgb = data['A'].type(FloatTensor)
    gen = G_AB(rgb)

    # 转换回1通道
    B_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

    gray = data['B']
    # save_image(inverse_normalize(B_transforms(gray),mean=(0.5, ), std=(0.5, )),output_path+'/B_img/'+str(i)+'.jpg')
    # save_image(inverse_normalize(B_transforms(gen),mean=(0.5, ), std=(0.5, )),output_path+'/fake_B_img/'+str(i)+'.jpg')
    save_image(rgb, output_path+'/A_img/'+str(i)+'.jpg')
    save_image(B_transforms(gray),output_path+'/B_img/'+str(i)+'.jpg')
    save_image(B_transforms(gen),output_path+'/fake_B_img/'+str(i)+'.jpg')   