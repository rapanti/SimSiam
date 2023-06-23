#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import builder
import utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def main(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print(f"git:\n  {utils.get_sha()}\n")
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    with (Path(args.output_dir) / "train.json").open("w") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    model = builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim
    ).cuda()

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    print(model)

    init_lr = args.lr * args.batch_size / 256

    criterion = nn.CosineSimilarity(dim=1).cuda()

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    fp16 = torch.cuda.amp.GradScaler() if args.fp16 else None

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16=fp16,
    )
    start_epoch = to_restore["epoch"]

    # ============ preparing data ... ============
    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    augmentation = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # do not use blur for CIFAR
        # transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    transform = TwoCropsTransform(transforms.Compose(augmentation))

    args.batch_size_per_gpu = args.batch_size // utils.get_world_size()
    dataset = datasets.CIFAR10(args.data_path, True, transform)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    log_dir = os.path.join(args.output_dir, "summary")
    board = SummaryWriter(log_dir) if utils.is_main_process() else None

    for epoch in range(start_epoch, args.epochs):
        loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        train(loader, model, criterion, optimizer, epoch, args, fp16, board)

        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'fp16': fp16.state_dict() if fp16 is not None else None,
        }
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))


def train(loader, model, criterion, optimizer, epoch, args, fp16, board):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        it = len(loader) * epoch + it  # global training iteration

        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16 is not None):
            p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        optimizer.zero_grad()
        if fp16 is None:
            loss.backward()
            optimizer.step()
        else:
            fp16.scale(loss).backward()
            fp16.step(optimizer)
            fp16.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if board is not None:
            board.add_scalar("loss", loss.item(), it)
            board.add_scalar("lr", optimizer.param_groups[0]["lr"], it)


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def get_args_parser():
    p = argparse.ArgumentParser("SimSiam", description='PyTorch ImageNet Training', add_help=False)
    p.add_argument('-a', '--arch', default='resnet18')
    p.add_argument('--epochs', default=800, type=int, help='number of total epochs to run')
    p.add_argument('-b', '--batch_size', default=512, type=int,
                   help='mini-batch size (default: 512), this is the total '
                        'batch size of all GPUs on the current node when')
    p.add_argument('--lr', default=0.03, type=float, help='initial (base) learning rate')
    p.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
    p.add_argument('--wd', '--weight_decay', default=5e-4, type=float, help='weight decay (default: 1e-4)',
                   dest="weight_decay")
    p.add_argument('--fp16', default=True, type=utils.bool_flag, help="Whether or not to use half precision for training.")

    # simsiam specific configs:
    p.add_argument('--dim', default=2048, type=int,
                   help='feature dimension (default: 2048)')
    p.add_argument('--pred_dim', default=512, type=int,
                   help='hidden dimension of the predictor (default: 512)')
    p.add_argument('--fix_pred_lr', default=False, type=utils.bool_flag,
                   help='Fix learning rate for the predictor')

    # Misc
    p.add_argument('--dataset', default="CIFAR10", type=str)
    p.add_argument('--data_path', type=str, help='path to training data.')
    p.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    p.add_argument('--saveckp_freq', default=0, type=int, help='Save checkpoint every x epochs.')
    p.add_argument('--seed', default=0, type=int, help='Random seed.')
    p.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    p.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    p.add_argument("--dist_backend", default="nccl", type=str, help="Distributed backend.")

    return p


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
