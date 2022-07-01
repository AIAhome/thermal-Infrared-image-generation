# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import shutil
import torch
import torch.distributed as dist

from models import *


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, is_best, output_dir):
    if is_main_process():
        ckpt_path = f'{output_dir}/checkpoint.pt'
        best_path = f'{output_dir}/checkpoint_best.pt'
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copyfile(ckpt_path, best_path)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def build_model(args):
    if args.model == 'resnet':
        shared_dim = args.dim * 2 ** args.n_downsample
        input_shape = (args.channels, args.img_height, args.img_width)

        shared_E = ResidualBlock(features=shared_dim).cuda(args.gpu)
        E1 = ResNetEncoder(dim=args.dim, n_downsample=args.n_downsample, shared_block=shared_E).cuda(args.gpu)
        E2 = ResNetEncoder(dim=args.dim, n_downsample=args.n_downsample, shared_block=shared_E).cuda(args.gpu)
        shared_G = ResidualBlock(features=shared_dim).cuda(args.gpu)
        G1 = ResNetGenerator(dim=args.dim, n_upsample=args.n_downsample, shared_block=shared_G).cuda(args.gpu)
        G2 = ResNetGenerator(dim=args.dim, n_upsample=args.n_downsample, shared_block=shared_G).cuda(args.gpu)
        D1 = ResNetDiscriminator(input_shape).cuda(args.gpu)
        D2 = ResNetDiscriminator(input_shape).cuda(args.gpu)
        if args.distributed:
            E1 = torch.nn.parallel.DistributedDataParallel(E1, device_ids=[args.gpu], bucket_cap_mb=200)
            E2 = torch.nn.parallel.DistributedDataParallel(E2, device_ids=[args.gpu], bucket_cap_mb=200)
            G1 = torch.nn.parallel.DistributedDataParallel(G1, device_ids=[args.gpu], bucket_cap_mb=200)
            G2 = torch.nn.parallel.DistributedDataParallel(G2, device_ids=[args.gpu], bucket_cap_mb=200)
            D1 = torch.nn.parallel.DistributedDataParallel(D1, device_ids=[args.gpu], bucket_cap_mb=200)
            D2 = torch.nn.parallel.DistributedDataParallel(D2, device_ids=[args.gpu], bucket_cap_mb=200)

    elif args.model == 'vit':
        dim = args.embed_dim
        num_heads = args.num_heads
        input_shape = (args.channels, args.img_height, args.img_width)

        shared_E = Block(dim=dim, num_heads=num_heads).cuda(args.gpu)
        E1 = ViTEncoder(shared_E, img_size=args.img_height, embed_dim=dim, num_heads=num_heads).cuda(args.gpu)
        E2 = ViTEncoder(shared_E, img_size=args.img_height, embed_dim=dim, num_heads=num_heads).cuda(args.gpu)
        shared_G = Block(dim=dim, num_heads=num_heads).cuda(args.gpu)
        G1 = ViTGenerator(shared_G, img_size=args.img_height, embed_dim=dim, num_heads=num_heads).cuda(args.gpu)
        G2 = ViTGenerator(shared_G, img_size=args.img_height, embed_dim=dim, num_heads=num_heads).cuda(args.gpu)
        D1 = ResNetDiscriminator(input_shape).cuda(args.gpu)
        D2 = ResNetDiscriminator(input_shape).cuda(args.gpu)
        if args.distributed:
            E1 = torch.nn.parallel.DistributedDataParallel(E1, device_ids=[args.gpu], bucket_cap_mb=200)
            E2 = torch.nn.parallel.DistributedDataParallel(E2, device_ids=[args.gpu], bucket_cap_mb=200)
            G1 = torch.nn.parallel.DistributedDataParallel(G1, device_ids=[args.gpu], bucket_cap_mb=200)
            G2 = torch.nn.parallel.DistributedDataParallel(G2, device_ids=[args.gpu], bucket_cap_mb=200)
            D1 = torch.nn.parallel.DistributedDataParallel(D1, device_ids=[args.gpu], bucket_cap_mb=200)
            D2 = torch.nn.parallel.DistributedDataParallel(D2, device_ids=[args.gpu], bucket_cap_mb=200)

    else:
        raise NotImplementedError

    return E1, E2, G1, G2, D1, D2