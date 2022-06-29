import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys


import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
import utils

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
## Training
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="FLIR_ADAS_v2", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=200, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
## Model
parser.add_argument("--model", type=str, default='vit', choices=['resnet', 'vit'])
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=20, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--embed_dim", type=int, default=768, help="number of embed dim of vit")
parser.add_argument("--num_heads", type=int, default=12, help="number of heads of vit")
## Distributed
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
args = parser.parse_args()
print(args)

utils.init_distributed_mode(args)
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

exp_name = 'vit'

# Losses
criterion_GAN = torch.nn.MSELoss().cuda(args.gpu)
criterion_pixel = torch.nn.L1Loss().cuda(args.gpu)

E1, E2, G1, G2, D1, D2 = utils.build_model(args)

if args.epoch != 0:
    # Load pretrained models
    E1.load_state_dict(torch.load("saved_models/%s/E1_%d.pth" % (args.dataset_name, args.epoch)))
    E2.load_state_dict(torch.load("saved_models/%s/E2_%d.pth" % (args.dataset_name, args.epoch)))
    G1.load_state_dict(torch.load("saved_models/%s/G1_%d.pth" % (args.dataset_name, args.epoch)))
    G2.load_state_dict(torch.load("saved_models/%s/G2_%d.pth" % (args.dataset_name, args.epoch)))
    D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (args.dataset_name, args.epoch)))
    D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (args.dataset_name, args.epoch)))
else:
    # Initialize weights
    E1.apply(weights_init_normal)
    E2.apply(weights_init_normal)
    G1.apply(weights_init_normal)
    G2.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)

# Loss weights
lambda_0 = 10  # GAN
lambda_1 = 0.1  # KL (encoded images)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 0.1  # KL (encoded translated images)
lambda_4 = 100  # Cycle pixel-wise

# Optimizers
if args.model == 'resnet':
    optimizer_G = torch.optim.Adam(
        itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),
        lr=args.lr,
        betas=(args.b1, args.b2),
    )
elif args.model == 'vit':
    optimizer_G = torch.optim.AdamW(
        itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),
        lr=args.lr,
        betas=(args.b1, args.b2),
    )

optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step
)

# Image transformations
transforms_ = [
    transforms.Resize(int(args.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((args.img_height, args.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Dataset
train_dataset = ImageDataset("/data1/shaoyuan/%s" % args.dataset_name, transforms_=transforms_, unaligned=True)
val_dataset = ImageDataset("/data1/shaoyuan/%s" % args.dataset_name, transforms_=transforms_, unaligned=True, mode="val")

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
else:
    train_sampler = None
    val_sampler = None


dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=(train_sampler is None),
    pin_memory=True,
    num_workers=args.n_cpu,
    sampler=train_sampler
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=5,
    shuffle=(val_sampler is None),
    num_workers=1,
    sampler=val_sampler
)


def sample_images(batches_done):
    if utils.is_main_process():
        with torch.no_grad():
            """Saves a generated sample from the test set"""
            imgs = next(iter(val_dataloader))
            X1 = torch.Tensor(imgs["A"]).cuda(args.gpu, non_blocking=True)
            X2 = torch.Tensor(imgs["B"]).cuda(args.gpu, non_blocking=True)
            _, Z1 = E1(X1)
            _, Z2 = E2(X2)
            fake_X1 = G1(Z2)
            fake_X2 = G2(Z1)
            img_sample = torch.cat((X1.data, fake_X2.data, X2.data, fake_X1.data), 0)
            save_image(img_sample, "images/%s/%s.png" % (exp_name, batches_done), nrow=5, normalize=True)


def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


# ----------
#  Training
# ----------

if not os.path.isdir('images/%s' % exp_name): os.mkdir('images/%s' % exp_name)
if not os.path.isdir('saved_models/%s' % exp_name): os.mkdir('saved_models/%s' % exp_name)

prev_time = time.time()
for epoch in range(args.epoch, args.n_epochs):
    if args.distributed:
        train_sampler.set_epoch(epoch)
    for i, batch in enumerate(dataloader):

        # Set model input
        X1 = torch.Tensor(batch["A"]).cuda(args.gpu, non_blocking=True)
        X2 = torch.Tensor(batch["B"]).cuda(args.gpu, non_blocking=True)

        # Adversarial ground truths
        if args.distributed:
            output_shape = D1.module.output_shape
        else:
            output_shape = D1.output_shape
        valid = torch.Tensor(np.ones((X1.size(0), *output_shape))).cuda(args.gpu, non_blocking=True)
        fake = torch.Tensor(np.zeros((X1.size(0), *output_shape))).cuda(args.gpu, non_blocking=True)
        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        mu1, Z1 = E1(X1)
        mu2, Z2 = E2(X2)

        # Reconstruct images
        recon_X1 = G1(Z1)
        recon_X2 = G2(Z2)

        # Translate images
        fake_X1 = G1(Z2)
        fake_X2 = G2(Z1)

        # Cycle translation
        mu1_, Z1_ = E1(fake_X1)
        mu2_, Z2_ = E2(fake_X2)
        cycle_X1 = G1(Z2_)
        cycle_X2 = G2(Z1_)

        # Losses
        loss_GAN_1 = lambda_0 * criterion_GAN(D1(fake_X1), valid)
        loss_GAN_2 = lambda_0 * criterion_GAN(D2(fake_X2), valid)
        loss_KL_1 = lambda_1 * compute_kl(mu1)
        loss_KL_2 = lambda_1 * compute_kl(mu2)
        loss_ID_1 = lambda_2 * criterion_pixel(recon_X1, X1)
        loss_ID_2 = lambda_2 * criterion_pixel(recon_X2, X2)
        loss_KL_1_ = lambda_3 * compute_kl(mu1_)
        loss_KL_2_ = lambda_3 * compute_kl(mu2_)
        loss_cyc_1 = lambda_4 * criterion_pixel(cycle_X1, X1)
        loss_cyc_2 = lambda_4 * criterion_pixel(cycle_X2, X2)

        # Total loss
        loss_G = (
            loss_KL_1
            + loss_KL_2
            + loss_ID_1
            + loss_ID_2
            + loss_GAN_1
            + loss_GAN_2
            + loss_KL_1_
            + loss_KL_2_
            + loss_cyc_1
            + loss_cyc_2
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = criterion_GAN(D1(X1), valid) + criterion_GAN(D1(fake_X1.detach()), fake)

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = criterion_GAN(D2(X2), valid) + criterion_GAN(D2(fake_X2.detach()), fake)

        loss_D2.backward()
        optimizer_D2.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = args.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, args.n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
        )

        # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    if args.checkpoint_interval != -1 and (epoch+1) % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(E1.module.state_dict(), "saved_models/%s/E1_%d.pth" % (exp_name, epoch))
        torch.save(E2.module.state_dict(), "saved_models/%s/E2_%d.pth" % (exp_name, epoch))
        torch.save(G1.module.state_dict(), "saved_models/%s/G1_%d.pth" % (exp_name, epoch))
        torch.save(G2.module.state_dict(), "saved_models/%s/G2_%d.pth" % (exp_name, epoch))
        torch.save(D1.module.state_dict(), "saved_models/%s/D1_%d.pth" % (exp_name, epoch))
        torch.save(D2.module.state_dict(), "saved_models/%s/D2_%d.pth" % (exp_name, epoch))