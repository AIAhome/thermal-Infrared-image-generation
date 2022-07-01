import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.utils import save_image


from tqdm import tqdm

from datasets import ImageDataset
from zmq import device
from models import *

import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

output_path = "/root/autodl-tmp/output/dualGAN"
data_path="/root/autodl-tmp/FLIR_ADAS_v2"
# "20220604_110426"
n_cpu = 8

timestamp_1 = '20220610_234847'#高还原网络
timestamp_2 = '20220610_181556'#迁移网络
epoch_1 = 10
epoch_2 = 10

############################################## keep the same as dualgan.py ################################################
G_AB_0 = Generator(3,3,is_A2B=True)
G_AB_0.load_state_dict(torch.load("%s/saved_models/%s/G_AB_%d.pth" % (output_path, timestamp_1, epoch_1)))
# G_AB_0 = torch.load("%s/saved_models/%s/G_AB_%d.pth" % (output_path, timestamp_1, epoch_1))
G_AB_1 = Generator(3,3,is_A2B=True)
G_AB_1.load_state_dict(torch.load("%s/saved_models/%s/G_AB_%d.pth" % (output_path, timestamp_2, epoch_2)))
# G_AB_1 = torch.load("%s/saved_models/%s/G_AB_%d.pth" % (output_path, timestamp_2, epoch_2))

# Configure dataloader
A_transforms = [
    # transforms.Grayscale(),    
    transforms.CenterCrop(size=1024),
    transforms.Resize(size=256),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
B_transforms = [
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(size=256),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("%s/images_rgb_train/data" % data_path, "%s/images_thermal_train/data" % data_path,
                A_transforms,B_transforms),
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

G_AB_1.to(device)
G_AB_1.eval()
for i,data in tqdm(enumerate(dataloader),total=len(dataloader)):
    rgb = data['A'].type(FloatTensor)
    gen = G_AB_1(rgb)

    gray = data['B']
    save_image(inverse_normalize(gray,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),output_path+'/B_img/'+str(i)+'.jpg')
    save_image(inverse_normalize(gen,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),output_path+'/fake_B_img/'+str(i)+'.jpg')
