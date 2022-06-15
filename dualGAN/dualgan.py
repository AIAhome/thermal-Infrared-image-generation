import argparse
import math
import os
import numpy as np
import itertools
import sys
import time
import json
from  datetime import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
from tqdm import tqdm

from datasets import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--data_path", type=str, help="path to the dataset")
parser.add_argument("--log_path",type=str,help="path to store the log")
parser.add_argument("--output_path",type=str,help="path to saved models and output images")
parser.add_argument("--g_lr", type=float, default=2e-3, help="adam: generator learning rate")
parser.add_argument('--d_lr',type=float,default=1e-4,help='adam: discriminator learning rate')
parser.add_argument("--alpha",type=float,default=0.9,help='alpha: smooth factor in RMSProp')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--a_channels", type=int, default=3, help="number of image channels")
parser.add_argument("--b_channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=3, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--lambda_adv",type=int,default=2,help='lambda_u,lambda_v in original paper, because lambda_u=lambda_v in paper\'s code')
parser.add_argument("--lambda_cycle",type=int,default=0.1,help="reconstruct loss weight in generator loss")
parser.add_argument('--lambda_interval',type=int,default=400,help='interval between the lambda change')
# parser.add_argument('--lambda_A',type=int,default=2000,help='coefficient w.r.t. lambda_cycle')
# parser.add_argument('--lambda_B',type=int,default=50,help="coefficient w.r.t. lambda_adv")
parser.add_argument('--warmup_epoches',type=int,default=1,help='number of epoches without adv loss')
parser.add_argument("--lambda_gp",type=int,default=10,help="gradient penalty loss weight in WGAN style discriminator")
parser.add_argument('--negative_ratio',type=float,default=0.01,help='the ratio in loss w.r.t. negative sample')
args = parser.parse_args()
print(args)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# timestamp = "20220608_141103"

os.makedirs("%s/images/%s" % (args.log_path,timestamp), exist_ok=True)
os.makedirs("%s/runs/%s" % (args.log_path,timestamp), exist_ok=True)
os.makedirs("%s/saved_models/%s" % (args.output_path,timestamp), exist_ok=True)
with open("%s/images/%s/config.json" % (args.log_path,timestamp),'wt') as fp:
    # fp.write(str(args))
    json.dump(vars(args),fp)

cuda = True if torch.cuda.is_available() else False

# Loss function
cycle_loss = torch.nn.L1Loss()
content_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
G_AB = Generator(args.a_channels,args.b_channels,is_A2B=True)
G_BA = Generator(args.b_channels,args.a_channels)
D_A = Discriminator(args.a_channels)
D_B = Discriminator(args.b_channels)
encoder = FeatureExtractor()
encoder.eval()


if cuda:
    G_AB.cuda()
    G_BA.cuda()
    D_A.cuda()
    D_B.cuda()
    encoder.cuda()
    cycle_loss.cuda()
    content_loss.cuda()

# if args.epoch != 0:
#     # Load pretrained models
#     G_AB.load_state_dict(torch.load("%s/saved_models/%s/G_AB_%d.pth" % (args.output_path, timestamp, args.epoch)))
#     G_BA.load_state_dict(torch.load("%s/saved_models/%s/G_BA_%d.pth" % (args.output_path, timestamp, args.epoch)))
#     D_A.load_state_dict(torch.load("%s/saved_models/%s/D_A_%d.pth" % (args.output_path,  timestamp, args.epoch)))
#     D_B.load_state_dict(torch.load("%s/saved_models/%s/D_B_%d.pth" % (args.output_path,  timestamp, args.epoch)))
# else:
#     # Initialize weights
#     G_AB.apply(weights_init_normal)
#     G_BA.apply(weights_init_normal)
#     D_A.apply(weights_init_normal)
#     D_B.apply(weights_init_normal)

# Load pretrained models
G_AB.load_state_dict(torch.load("%s/saved_models/%s/G_AB_%d.pth" % (args.output_path, "20220608_141103", 70)))
G_BA.load_state_dict(torch.load("%s/saved_models/%s/G_BA_%d.pth" % (args.output_path, "20220608_141103", 70)))
# Initialize weights
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)


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

train_dataloader = DataLoader(
    ImageDataset("%s/images_rgb_train/data" % args.data_path, "%s/images_thermal_train/data" % args.data_path,
                A_transforms,B_transforms),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.n_cpu,
)
val_dataloader = DataLoader(
    ImageDataset("%s/images_rgb_val/data" % args.data_path, "%s/images_thermal_val/data" % args.data_path,
                A_transforms,B_transforms),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)

# Optimizers
# optimizer_G = torch.optim.RMSprop(
#     itertools.chain(G_AB.parameters(), G_BA.parameters()),lr=args.g_lr,alpha=args.alpha
# )
optimizer_D_A = torch.optim.RMSprop(D_A.parameters(),lr=args.d_lr,alpha=args.alpha)
optimizer_D_B = torch.optim.RMSprop(D_B.parameters(),lr=args.d_lr,alpha=args.alpha)
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()),lr=args.g_lr,betas=[args.b1,args.b2]
)
# optimizer_D_A = torch.optim.Adam(D_A.parameters(),lr=args.d_lr,betas=[args.b1,args.b2])
# optimizer_D_B = torch.optim.Adam(D_B.parameters(),lr=args.d_lr,betas=[args.b1,args.b2])


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    validity = D.get_raw_output(interpolates)
    fake = Variable(FloatTensor(np.ones(validity.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=validity,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_images(batches_done):
    """Saves a generated sample from the val set"""
    def inverse_normalize(tensor, mean, std):# 避免normalize之后显示出来的图像偏黑
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    imgs = next(iter(val_dataloader))

    real_A = Variable(imgs["A"].type(FloatTensor))
    real_B = Variable(imgs["B"].type(FloatTensor))
    fake_A = G_BA(real_B)
    fake_B = G_AB(real_A)
    recov_A = G_BA(fake_B)
    recov_B = G_AB(fake_A)

    ABA = torch.cat((real_A.data, fake_B.data,recov_A.data), -2)
    BAB = torch.cat((real_B.data, fake_A.data,recov_B.data), -2)
    img_sample = torch.cat((ABA, BAB), 0)
    save_image(inverse_normalize(tensor=img_sample, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), "%s/images/%s/%s.png" % (args.log_path, timestamp, batches_done), nrow=8, normalize=True)
    # save_image(img_sample, "%s/images/%s/%s.png" % (args.log_path, timestamp, batches_done), nrow=8)


# ----------
#  Training
# ----------
writer= SummaryWriter('%s/runs/%s'%(args.log_path,timestamp))
batches_done = 0
t = 0
for epoch in range(args.n_epochs):
    epoch += args.epoch
    print("[epoch]:",str(epoch))
    for i, batch in tqdm(enumerate(train_dataloader),desc="batches training",total=len(train_dataloader)):

        # Configure input
        imgs_A = Variable(batch["A"].type(FloatTensor))
        imgs_B = Variable(batch["B"].type(FloatTensor))

        # ----------------------
        #  Train Discriminators
        # ----------------------

        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        # Generate a batch of images
        fake_A = G_BA(imgs_B).detach()
        fake_B = G_AB(imgs_A).detach()
        # negative_A = 255 - imgs_B
        # negative_B = 255 - imgs_A

        # ----------
        # Domain A
        # ----------

        # Compute gradient penalty for improved wasserstein training
        gp_A = compute_gradient_penalty(D_A, imgs_A.data, fake_A.data)
        # Adversarial loss
        if np.random.random() < 0.7:
            # D_A_loss = -torch.mean(D_A(imgs_A,fake_A)) + args.negative_ratio*torch.mean(D_A.get_raw_output(negative_A)) + args.lambda_gp * gp_A 
            D_A_loss = -torch.mean(D_A(imgs_A,fake_A)) + args.lambda_gp * gp_A 
        else:
            D_A_loss = -torch.mean(D_A(fake_A,imgs_A)) + args.lambda_gp * gp_A 

        # ----------
        # Domain B
        # ----------

        # Compute gradient penalty for improved wasserstein training
        gp_B = compute_gradient_penalty(D_B, imgs_B.data, fake_B.data)
        # Adversarial loss
        if np.random.random() < 0.7:
            # D_B_loss = -torch.mean(D_B(imgs_B,fake_B)) + args.negative_ratio*torch.mean(D_B.get_raw_output(negative_B))  + args.lambda_gp * gp_B
            D_B_loss = -torch.mean(D_B(imgs_B,fake_B)) + args.lambda_gp * gp_B
        else:
            D_B_loss = -torch.mean(D_B(fake_B,imgs_B)) + args.lambda_gp * gp_B
        
        # Total loss
        D_loss = D_A_loss + D_B_loss

        D_loss.backward()
        optimizer_D_A.step()
        optimizer_D_B.step()

        if i % args.n_critic == 0:

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Translate images to opposite domain
            fake_A = G_BA(imgs_B)
            fake_B = G_AB(imgs_A)

            # Reconstruct images
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)

            # # change the lambda_adv and lambda_cycle
            # if batches_done%args.lambda_interval == 0:
            #     lambda_adv = args.lambda_adv - math.exp(-0.5*t)*args.lambda_B
            #     lambda_cycle = args.lambda_cycle + math.exp(-0.5*t)*args.lambda_A
            #     t += 1
            if epoch < args.warmup_epoches:
                lambda_adv = 0
                G_adv = 0
            else:
                lambda_adv = args.lambda_adv
                # Adversarial loss
                G_adv = torch.mean(D_A(imgs_A,fake_A))\
                        + torch.mean(D_B(imgs_B,fake_B))\
                        # - torch.mean(D_A(fake_A,negative_A))\
                        # - torch.mean(D_B(fake_B,negative_B))
            
            # Cycle loss, i.e. L1 loss in paper
            G_cycle = cycle_loss(recov_A, imgs_A) + cycle_loss(recov_B, imgs_B)
            # content loss just like esrGAN
            # emb_fake_B = encoder(fake_B)
            # emb_imgs_A = encoder(imgs_A).detach()
            # emb_fake_A = encoder(fake_A)
            # emb_imgs_B = encoder(imgs_B).detach()

            emb_recov_B = encoder(recov_B)
            emb_imgs_A = encoder(imgs_A).detach()
            emb_recov_A = encoder(recov_A)
            emb_imgs_B = encoder(imgs_B).detach()

            # Total loss
            G_loss = lambda_adv * G_adv + args.lambda_cycle * G_cycle \
                    + args.lambda_cycle*0.5*content_loss(emb_recov_A,emb_imgs_A) + args.lambda_cycle*0.5*content_loss(emb_recov_B,emb_imgs_B)

            G_loss.backward()
            optimizer_G.step()

            # --------------
            # Log Progress
            # --------------

            # Determine approximate time left
            # batches_left = args.n_epochs * len(dataloader) - batches_done
            # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / args.n_critic)
            # prev_time = time.time()

            # sys.stdout.write(
            #     "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, cycle loss: %f] ETA: %s"
            #     % (
            #         epoch,
            #         args.n_epochs,
            #         i,
            #         len(dataloader),
            #         D_loss.item(),
            #         G_adv.data.item(),
            #         G_cycle.item(),
            #         time_left,
            #     )
            # )

        # Check sample interval => save sample if there
        if batches_done % args.sample_interval == 0:
            sample_images(batches_done)

        batches_done += 1
    
        # Log per batch
        if epoch==0 and i==0:
            # writer.add_graph(D_A,(imgs_A,imgs_A))
            # writer.add_graph(D_B,(imgs_B,imgs_b))
            # writer.add_graph(G_BA,imgs_B)
            writer.add_graph(G_AB,imgs_A)
        writer.add_scalars('loss',
                        { 'D_loss' : D_loss,"G_loss":G_loss},
                        epoch * len(train_dataloader) + i)
        # writer.add_scalar('Generator loss',
        #                 {'adv_loss':G_adv,'cycle_loss':G_cycle},
        #                 epoch * len(train_dataloader) + i)
    writer.flush()
    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "%s/saved_models/%s/G_AB_%d.pth" % (args.output_path,timestamp, epoch))
        torch.save(G_BA.state_dict(), "%s/saved_models/%s/G_BA_%d.pth" % (args.output_path,timestamp, epoch))
        torch.save(D_A.state_dict(), "%s/saved_models/%s/D_A_%d.pth" % (args.output_path,timestamp, epoch))
        torch.save(D_B.state_dict(), "%s/saved_models/%s/D_B_%d.pth" % (args.output_path,timestamp, epoch))
        import requests
        resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                            json={
                                "token": "81be739b24fd",
                                "title": "dualGAN实验",
                                "name": "50个epoch小实验",
                                "content": "finish %s epoch"% epoch
                            })
        print(resp.content.decode())
