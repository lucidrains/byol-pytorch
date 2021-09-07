#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# code taken from https://github.com/facebookresearch/simsiam

import builtins
import math
import os
import pathlib
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import yaml

from metassl.utils.data import get_train_valid_loader
from metassl.utils.supporter import Supporter
from utils.simsiam import SimSiam
from utils.meters import AverageMeter, ProgressMeter

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
    )


def main(config, expt_dir):
    # args = parser.parse_args()
    
    if config.expt.seed is not None:
        random.seed(config.expt.seed)
        torch.manual_seed(config.expt.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
            )
    
    if config.expt.gpu is not None:
        warnings.warn(
            'You have chosen a specific GPU. This will completely '
            'disable data parallelism.'
            )
    
    if config.expt.dist_url == "env://" and config.expt.world_size == -1:
        config.expt.world_size = int(os.environ["WORLD_SIZE"])
    
    config.expt.distributed = config.expt.world_size > 1 or config.expt.multiprocessing_distributed
    
    ngpus_per_node = torch.cuda.device_count()
    if config.expt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.expt.world_size = ngpus_per_node * config.expt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, expt_dir))
    else:
        # Simply call main_worker function
        main_worker(config.expt.gpu, ngpus_per_node, config, expt_dir)


def main_worker(gpu, ngpus_per_node, config, expt_dir):
    config.expt.gpu = gpu
    
    # suppress printing if not master
    if config.expt.multiprocessing_distributed and config.expt.gpu != 0:
        def print_pass(*args):
            pass
        
        builtins.print = print_pass
    
    if config.expt.gpu is not None:
        print("Use GPU: {} for training".format(config.expt.gpu))
    
    if config.expt.distributed:
        if config.expt.dist_url == "env://" and config.expt.rank == -1:
            config.expt.rank = int(os.environ["RANK"])
        if config.expt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.expt.rank = config.expt.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=config.expt.dist_backend, init_method=config.expt.dist_url,
            world_size=config.expt.world_size, rank=config.expt.rank
            )
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(config.model.model_type))
    model = SimSiam(
        models.__dict__[config.model.model_type],
        config.simsiam.dim, config.simsiam.pred_dim
        )
    
    # infer learning rate before changing batch size
    init_lr = config.optim.lr_high * config.train.batch_size / 256
    
    if config.expt.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.expt.gpu is not None:
            torch.cuda.set_device(config.expt.gpu)
            model.cuda(config.expt.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.train.batch_size = int(config.train.batch_size / ngpus_per_node)
            config.workers = int((config.expt.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.expt.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.expt.gpu is not None:
        torch.cuda.set_device(config.expt.gpu)
        model = model.cuda(config.expt.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model) # print model after SyncBatchNorm
    
    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(config.expt.gpu)
    
    if config.simsiam.fix_pred_lr:
        optim_params = [{
            'params': model.module.encoder.parameters(),
            'fix_lr': False
            },
            {
                'params': model.module.predictor.parameters(),
                'fix_lr': True
                }]
    else:
        optim_params = model.parameters()
    
    optimizer = torch.optim.SGD(
        optim_params, init_lr,
        momentum=config.optim.momentum,
        weight_decay=config.optim.weight_decay
        )
    
    # optionally resume from a checkpoint
    if config.expt.resume:
        if os.path.isfile(config.expt.resume):
            print("=> loading checkpoint '{}'".format(config.expt.resume))
            if config.expt.gpu is None:
                checkpoint = torch.load(config.expt.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.expt.gpu)
                checkpoint = torch.load(config.expt.resume, map_location=loc)
            config.expt.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(
                "=> loaded checkpoint '{}' (epoch {})"
                    .format(config.expt.resume, checkpoint['epoch'])
                )
        else:
            print("=> no checkpoint found at '{}'".format(config.expt.resume))
    
    cudnn.benchmark = True
    
    # Data loading code
    traindir = os.path.join(config.data.dataset, 'train')
    
    train_loader, valid_loader, train_sampler, valid_sampler = get_train_valid_loader(
        data_dir=traindir,
        batch_size=config.train.batch_size,
        random_seed=config.expt.seed,
        dataset_name="ImageNet",
        shuffle=True,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=True,
        distributed=config.expt.distributed,
        drop_last=True,
        )
    
    for epoch in range(config.train.start_epoch, config.train.epochs):
        if config.expt.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, config)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config, expt_dir)
        
        if not config.expt.multiprocessing_distributed or (config.expt.multiprocessing_distributed
                                                           and config.expt.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch':      epoch + 1,
                    'arch':       config.model.model_type,
                    'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(expt_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                )


def train(train_loader, model, criterion, optimizer, epoch, config, expt_dir):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
        )
    
    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if config.expt.gpu is not None:
            images[0] = images[0].cuda(config.expt.gpu, non_blocking=True)
            images[1] = images[1].cuda(config.expt.gpu, non_blocking=True)
        
        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        
        losses.update(loss.item(), images[0].size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % config.expt.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, init_lr, epoch, config):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / config.train.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    user = os.environ.get('USER')
    
    with open("metassl/default_metassl_config.yaml", "r") as f:
        config = yaml.load(f)
    
    expt_dir = f"/home/{user}/workspace/experiments/metassl"
    
    expt_dir = pathlib.Path(expt_dir)
    
    config['data']['data_dir'] = f'/home/{user}/workspace/data/metassl'
    
    supporter = Supporter(experiments_dir=expt_dir, config_dict=config, count_expt=True)
    config = supporter.get_config()
    
    main(config=config, expt_dir=expt_dir)
