import argparse
import json
import math
import os
import random
from pathlib import Path

import einops
import torch
import torch.nn as nn
import torch.nn.functional as nnf
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

    # summary(model, [(3, 224, 224), (3, 224, 224)])

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
    transform = TwoCropsTransform(args.num_crops)

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

    select_views = select_names[args.select_fn]

    log_dir = os.path.join(args.output_dir, "summary")
    board = SummaryWriter(log_dir) if utils.is_main_process() else None

    for epoch in range(start_epoch, args.epochs):
        loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        train(loader, model, criterion, optimizer, epoch, args, fp16, board, select_views)

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


def train(loader, model, criterion, optimizer, epoch, args, fp16, board, select_views):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        it = len(loader) * epoch + it  # global training iteration

        images = [im.cuda(non_blocking=True) for im in images]
        # x1, x2, y0 = images
        # b, h, w, c = x1.shape

        x1, x2 = select_views(images, model, fp16)

        # y0 = select_patches(x1, y0, model)
        #
        # if rc_prob > 0:
        #     random_values = torch.rand(b, device=x1.device)
        #     use_random = random_values < rc_prob
        #     x2 = torch.where(use_random[:, None, None, None], x2, y0)
        # else:
        #     x2 = y0

        with torch.cuda.amp.autocast(fp16 is not None):
            p1, p2, z1, z2 = model(x1=x1, x2=x2)
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

    def __init__(self, num_crops=2):
        self.num_crops = num_crops
        rrc = [transforms.RandomResizedCrop(32, (0.2, 1))]
        augmentation = [
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # do not use blur for blur
            # transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
        self.base_transform = transforms.Compose(rrc + augmentation)
        self.color_transform = transforms.Compose(augmentation)

    def __call__(self, x):
        # return [self.base_transform(x), self.base_transform(x), self.color_transform(x)]
        return [self.base_transform(x) for _ in range(self.num_crops)]


@torch.no_grad()
def select_views_cross(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.get_proj(torch.cat(images, dim=0))
    p1s, z1s = model_out[0].chunk(len(images)), model_out[1].chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n in range(len(images)):
        p1, z1 = p1s[n], z1s[n]
        for m in range(n + 1, len(images)):
            p2, z2 = p1s[m], z1s[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(p1, z2) + nnf.cosine_similarity(p2, z1)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_views_avgpool(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.encoder.avgpool_activations(torch.cat(images, dim=0))
    embeds = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(embeds):
        e1 = embeds[n]
        for m in range(n + 1, len(embeds)):
            e2 = embeds[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_views_linear(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.encoder.first_linear_activations(torch.cat(images, dim=0))
    embeds = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(embeds):
        e1 = embeds[n]
        for m in range(n + 1, len(embeds)):
            e2 = embeds[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_views_first_layer(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        embeds = [model.module.encoder.first_layer_activations(img).view(b, -1) for img in images]
    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)
    for n, x in enumerate(embeds):
        a1 = embeds[n]
        for m in range(n + 1, len(embeds)):
            a2 = embeds[m]
            sim = nnf.cosine_similarity(a1, a2)
            score, indices = torch.stack((score, sim)).min(dim=0)
            indices = indices.type(torch.bool)
            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)
    return out1, out2


@torch.no_grad()
def select_views_anchor(images, model, fp16):
    embeds = [model.module.get_proj(img) for img in images]
    p1, z1 = embeds[0]
    scores = [nnf.cosine_similarity(p1, z2) + nnf.cosine_similarity(p2, z1) for p2, z2 in embeds[1:]]
    stacked = torch.stack(scores)
    values, indic = stacked.min(dim=0)
    a = nnf.one_hot(indic, len(scores)).T
    out = torch.zeros_like(images[0])
    for n, img in enumerate(images[1:]):
        out += a[n].view(-1, 1, 1, 1) * img
    return images[0], out


select_names = {
    "anchor": select_views_anchor,
    "cross": select_views_cross,
    "firstlayer": select_views_first_layer,
    "avgpool": select_views_avgpool,
    "linear": select_views_linear,
}


@torch.no_grad()
def select_views_cross_(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        embeds = [model.module.get_proj(img) for img in images]

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)
    for n, x in enumerate(embeds):
        p1, z1 = embeds[n]
        for m in range(n + 1, len(embeds)):
            p2, z2 = embeds[m]
            sim = nnf.cosine_similarity(p1, z2) + nnf.cosine_similarity(p2, z1)
            score, indices = torch.stack((score, sim)).min(dim=0)
            indices = indices.type(torch.bool)
            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)
    return out1, out2


# Select-Patches with unfold operation
@torch.no_grad()
def select_patches(x1, x2, model):
    b, c, height, width = x1.shape

    target = model.module.get_proj(x1)

    _, _, h, w = get_params(x2, (0.2, 1))
    unfolded = nnf.unfold(x2, (h, w), stride=4)
    unfolded = einops.rearrange(unfolded, "b (c h w) n -> (b n) c h w", c=c, h=h, w=w)
    samples = nnf.interpolate(unfolded, (32, 32), mode='bilinear', antialias=True)
    embeds = model.module.get_proj(samples)
    out = process_patches(target, samples, embeds)

    return out


def process_patches(target, samples, embeds):
    p1, z1 = target
    p2, z2 = embeds

    b = p1.size(0)
    bs = samples.size(0)
    n = bs // b
    p1 = p1.repeat(1, n)
    z1 = z1.repeat(1, n)
    p1 = einops.rearrange(p1, "b (n d) -> (b n) d", n=n)
    z1 = einops.rearrange(z1, "b (n d) -> (b n) d", n=n)

    # loss = nnf.cosine_similarity(student_out, tmp, dim=-1)
    loss = nnf.cosine_similarity(p1, z2) + nnf.cosine_similarity(p2, z1)
    loss = einops.rearrange(loss, "(b n) -> b n", n=n)
    indices = nnf.one_hot(loss.argmin(dim=-1), num_classes=n).type(samples.dtype)
    samples = einops.rearrange(samples, "(b n) c h w -> b n c h w", n=n)
    patches = torch.einsum("b n, b n c h w -> b c h w", indices, samples)
    return patches


def get_params(img, scale, ratio=(3.0 / 4.0, 4.0 / 3.0)):
    batch, channels, height, width = img.shape
    area = height * width

    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            return i, j, h, w

    # Fallback maximum patch
    scale_sqrt = int(math.sqrt(scale[1]))
    h = scale_sqrt * height
    w = scale_sqrt * width
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def get_args_parser():
    p = argparse.ArgumentParser("SimSiam", description='PyTorch ImageNet Training', add_help=False)
    p.add_argument('-a', '--arch', default='resnet18')
    p.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    p.add_argument('--num_crops', type=int, default=4, help="Number of crops.")
    p.add_argument('--select_fn', type=str, default="avgpool", choices=select_names)
    p.add_argument('-b', '--batch_size', default=512, type=int,
                   help='mini-batch size (default: 512), this is the total '
                        'batch size of all GPUs on the current node when')
    p.add_argument('--lr', default=0.03, type=float, help='initial (base) learning rate')
    p.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
    p.add_argument('--wd', '--weight_decay', default=5e-4, type=float, help='weight decay (default: 1e-4)',
                   dest="weight_decay")
    p.add_argument('--fp16', default=True, type=utils.bool_flag,
                   help="Whether or not to use half precision for training.")

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
