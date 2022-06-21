import argparse
import os
from re import A
import numpy as np
import math
import itertools
import sys
import datetime
import time
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
# from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn   
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--epoch",
                    type=int,
                    default=0,
                    help="epoch to start training from")
parser.add_argument("--n_epochs",
                    type=int,
                    default=101,
                    help="number of epochs of training")
parser.add_argument('--data_path',
                    type=str,
                    default='./data',
                    help=' path to the datasets')
parser.add_argument('--output_path',
                    type=str,
                    default='./output',
                    help='path to save model and output images')
parser.add_argument('--log_path',
                    type=str,
                    default='./log',
                    help='path to store the logs')
parser.add_argument("--batch_size",
                    type=int,
                    default=4,
                    help="size of the batches")
parser.add_argument("--lr",
                    type=float,
                    default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1",
                    type=float,
                    default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2",
                    type=float,
                    default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation")
parser.add_argument(
    '--a_channels', type=int, default=3,
    help='channels of the dataset_a')  # 原始rgb图像为channels=3, 如果转成灰度图就是1
parser.add_argument('--b_channels',
                    type=int,
                    default=1,
                    help='channels of the dataset_b')
parser.add_argument("--img_height",
                    type=int,
                    default=256,
                    help="size of image height")
parser.add_argument("--img_width",
                    type=int,
                    default=256,
                    help="size of image width")
parser.add_argument("--sample_interval",
                    type=int,
                    default=800,
                    help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval",
                    type=int,
                    default=-1,
                    help="interval between model checkpoints")
parser.add_argument("--gray",
                    type=bool,
                    default=False,
                    help='whether to gray the input image')
parser.add_argument("--lambda_adv",
                    type=float,
                    default=1,
                    help="adversarial loss weight")
parser.add_argument("--lambda_cycle",
                    type=float,
                    default=1,
                    help="cycle loss weight")
parser.add_argument("--lambda_perceptual",
                    type=float,
                    default=0.05,
                    help="identity loss weight")
parser.add_argument("--lambda_content",
                    type=float,
                    default=1,
                    help="content loss weight")
parser.add_argument("--warmup_epoch",
                    type=int,
                    default=1,
                    help="number of epoches only cycle and content")
args = parser.parse_args()
print(args)

# Create sample, checkpoint directories and logs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("%s/images/%s" % (args.output_path, timestamp), exist_ok=True)
os.makedirs("%s/saved_models/%s" % (args.output_path, timestamp),
            exist_ok=True)
os.makedirs("%s/runs/%s" % (args.log_path, timestamp), exist_ok=True)

# Losses
adversarial_loss = torch.nn.MSELoss()
cycle_loss = torch.nn.L1Loss()
perceptual_loss = torch.nn.L1Loss()
content_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
a_input_shape = (args.a_channels, args.img_height, args.img_width)
b_input_shape = (args.b_channels, args.img_height, args.img_width)
G_AB = GeneratorUNet(args.a_channels, args.b_channels)
G_BA = GeneratorUNet(args.b_channels, args.a_channels)
D_A = Discriminator(a_input_shape)
D_B = Discriminator(b_input_shape)
vgg_feature_extractor = FeatureExtractor()

# gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G_AB = G_AB.to(device)
G_BA = G_BA.to(device)
D_A = D_A.to(device)
D_B = D_B.to(device)
adversarial_loss.to(device)
cycle_loss.to(device)
perceptual_loss.to(device)
content_loss.to(device)
vgg_feature_extractor.to(device)

# Set feature extractor to inference mode
vgg_feature_extractor.eval()

if args.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(
        torch.load("%s/saved_models/%s/G_AB_%d.pth" %
                   (args.output_path, timestamp, args.epoch)))
    G_BA.load_state_dict(
        torch.load("%s/saved_models/%s/G_BA_%d.pth" %
                   (args.output_path, timestamp, args.epoch)))
    D_A.load_state_dict(
        torch.load("%s/saved_models/%s/D_A_%d.pth" %
                   (args.output_path, timestamp, args.epoch)))
    D_B.load_state_dict(
        torch.load("%s/saved_models/%s/D_B_%d.pth" %
                   (args.output_path, timestamp, args.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(),
                                               G_BA.parameters()),
                               lr=args.lr,
                               betas=(args.b1, args.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(),
                                 lr=args.lr,
                                 betas=(args.b1, args.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(),
                                 lr=args.lr,
                                 betas=(args.b1, args.b2))


# Dataset loader
A_transforms = [
    # transforms.Grayscale(),
    transforms.CenterCrop(size=1024),
    transforms.Resize(size=args.img_height, interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
]
B_transforms = [
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(size=args.img_height),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
]

dataloader = DataLoader(
    ImageDataset("%s/images_rgb_train/data" % args.data_path,
                 "%s/images_thermal_train/data" % args.data_path,
                 A_transforms=A_transforms,
                 B_transfroms=B_transforms),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.n_cpu,
)
val_dataloader = DataLoader(
    ImageDataset("%s/images_rgb_val/data" % args.data_path,
                 "%s/images_thermal_val/data" % args.data_path,
                 A_transforms=A_transforms,
                 B_transfroms=B_transforms),
    batch_size=5,
    shuffle=True,
    num_workers=args.n_cpu,
)

# sample image to save
def sample_images(batches_done, device):
    """Saves a generated sample from the validation set"""
    nrow = 5
    imgs = next(iter(val_dataloader))
    eval_B_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
    G_AB.eval()
    G_BA.eval()
    real_A = imgs['A'].to(device)
    fake_B = G_AB(real_A)
    cycle_A = G_BA(fake_B)
    real_B = imgs['B'].to(device)
    fake_A = G_BA(real_B)
    cycle_B = G_AB(fake_A)

    # 红外图的通道重设为1
    gray_fake_B = eval_B_transforms(fake_B)
    gray_cycle_B = eval_B_transforms(cycle_B)

    real_A = make_grid(real_A, nrow=nrow, normalize=True)  # 拼成一副图像
    real_B = make_grid(real_B, nrow=nrow, normalize=True)
    fake_A = make_grid(fake_A, nrow=nrow, normalize=True)
    fake_B = make_grid(fake_B, nrow=nrow, normalize=True)
    gray_fake_B = make_grid(gray_fake_B, nrow=nrow, normalize=True)
    cycle_A = make_grid(cycle_A, nrow=nrow, normalize=True)
    gray_cycle_B = make_grid(gray_cycle_B, nrow=nrow, normalize=True)
    img_sample = torch.cat((real_A.data, fake_B.data, gray_fake_B.data, cycle_A.data,
                            real_B.data, fake_A.data, gray_cycle_B.data),
                           1)  # 上下拼接(0是channel)
    save_image(img_sample,
               "%s/images/%s/%s.png" %
               (args.output_path, timestamp, batches_done),
               normalize=True)


# ----------
#  Training
# ----------

writer = SummaryWriter('%s/runs/%s' % (args.log_path, timestamp))
lambda_adv = args.lambda_adv
lambda_cycle = args.lambda_cycle
lambda_perceptual = args.lambda_perceptual
lambda_content = args.lambda_content

for epoch in tqdm(range(args.epoch, args.n_epochs), desc='epoch', position=1):
    for i, batch in enumerate(
            tqdm(dataloader, desc='batch', position=0, colour='green')):

        # Model inputs
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        real_A.requires_grad = True
        real_B.requires_grad = True

        # Adversarial ground truths
        valid = torch.ones((real_A.size(0), *D_A.output_shape),
                           requires_grad=False).to(
                               device)  # D输出的是PatchGAN, 为N*N的矩阵每个元素的值代表其感受野真假
        fake = torch.zeros((real_A.size(0), *D_A.output_shape),
                           requires_grad=False).to(device)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = adversarial_loss(
            D_B(fake_B), valid)  # 与全1矩阵求MSEloss, D输出判断真假(每个元素的值代表其感受野为真的概率)
        fake_A = G_BA(real_B)
        loss_GAN_BA = adversarial_loss(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2  # 生成器loss, 即判别器输出概率与真的差值

        # Cycle loss
        loss_cycle_A = cycle_loss(G_BA(fake_B), real_A)
        loss_cycle_B = cycle_loss(G_AB(fake_A), real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Content loss
        cycle_A_features = vgg_feature_extractor(G_BA(fake_B))
        real_A_features = vgg_feature_extractor(real_A).detach()
        cycle_B_features = vgg_feature_extractor(G_AB(fake_A))
        real_B_features = vgg_feature_extractor(real_B).detach()
        loss_content_A = content_loss(cycle_A_features, real_A_features)
        loss_content_B = content_loss(cycle_B_features, real_B_features)
        loss_content = (loss_content_A + loss_content_B) / 2

        # Perceptual loss
        fake_A_features = vgg_feature_extractor(fake_A)
        fake_B_features = vgg_feature_extractor(fake_B)
        loss_perceptual_A = perceptual_loss(fake_A_features, real_A_features)
        loss_perceptual_B = perceptual_loss(fake_B_features, real_B_features)
        loss_perceptual = (loss_perceptual_A + loss_perceptual_B) / 2

        # Pre-train only cycle loss and content loss
        if epoch < args.warmup_epoch:
            lambda_adv = 0
            lambda_perceptual = 0
        else:
            lambda_adv = args.lambda_adv
            lambda_perceptual = args.lambda_perceptual

        # Total loss
        loss_G = lambda_adv * loss_GAN + lambda_cycle * loss_cycle + lambda_perceptual * loss_perceptual + lambda_content * loss_content  # 删去了pixelwise loss # TODO: 调参比例

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = adversarial_loss(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = adversarial_loss(D_A(fake_A.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake
                    ) / 2  # D需要real_loss, fake_loss均小, 即real判定接近1, fake判定接近0

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()
        # Real loss
        loss_real = adversarial_loss(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = adversarial_loss(D_B(fake_B.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = 0.5 * (loss_D_A + loss_D_B)

        # tensorboard print loss G&D
        writer.add_scalars('Loss', {
            'G_loss': loss_G,
            'D_loss': loss_D
        },
                           global_step=epoch * len(dataloader) + i)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = args.n_epochs * len(dataloader) - batches_done
        prev_time = time.time()

        # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(batches_done, device)

        # add model graph
        if epoch == 0 and i == 0:
            # writer.add_graph(G_AB, real_A)
            # writer.add_graph(G_BA, real_B)
            writer.add_graph(D_A, real_A)
            writer.add_graph(D_B, real_B)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(
            G_AB.state_dict(), "%s/saved_models/%s/G_AB_%d.pth" %
            (args.output_path, timestamp, epoch))
        torch.save(
            G_BA.state_dict(), "%s/saved_models/%s/G_BA_%d.pth" %
            (args.output_path, timestamp, epoch))
        torch.save(
            D_A.state_dict(), "%s/saved_models/%s/D_A_%d.pth" %
            (args.output_path, timestamp, epoch))
        torch.save(
            D_B.state_dict(), "%s/saved_models/%s/D_B_%d.pth" %
            (args.output_path, timestamp, epoch))

writer.close()
