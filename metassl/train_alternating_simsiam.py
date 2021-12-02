# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# code taken from https://github.com/facebookresearch/simsiam
import argparse
import builtins
import math
import os
import pathlib
import random
import time
import warnings

from collections import OrderedDict

import jsonargparse
import numpy as np
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
from jsonargparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.torch_utils import get_newest_model, check_and_save_checkpoint, deactivate_bn

warnings.filterwarnings("ignore", category=UserWarning)

try:
    # For execution in PyCharm
    from metassl.utils.data import get_train_valid_loader, get_test_loader
    from metassl.utils.config import AttrDict
    from metassl.utils.meters import AverageMeter, ProgressMeter, ExponentialMovingAverageMeter
    from metassl.utils.simsiam_alternating import SimSiam
    import metassl.models.resnet_cifar as our_cifar_resnets
except ImportError:
    # For execution in command line
    from .utils.data import get_train_valid_loader, get_test_loader
    from .utils.config import AttrDict
    from .utils.meters import AverageMeter, ProgressMeter, ExponentialMovingAverageMeter
    from .utils.simsiam_alternating import SimSiam
    from .models import resnet_cifar as our_cifar_resnets

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
    )


def main(config, expt_dir):
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
        print(f"Use GPU: {config.expt.gpu} for training")
    
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
    print(f"=> creating model '{config.model.model_type}'")
    if config.data.dataset == 'CIFAR10':
        # Use model from our model folder instead from torchvision!
        model = SimSiam(our_cifar_resnets.resnet18, config.simsiam.dim, config.simsiam.pred_dim)
    else:
        model = SimSiam(models.__dict__[config.model.model_type], config.simsiam.dim, config.simsiam.pred_dim)
    
    if config.model.turn_off_bn:
        print("Turning off BatchNorm in entire model.")
        deactivate_bn(model)
        model.encoder_head[6].bias.requires_grad = True
    
    # infer learning rate before changing batch size
    init_lr_pt = config.train.lr * config.train.batch_size / 256
    init_lr_ft = config.finetuning.lr * config.finetuning.batch_size / 256
    
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
            config.finetuning.batch_size = int(config.finetuning.batch_size / ngpus_per_node)
            config.train.batch_size = int(config.train.batch_size / ngpus_per_node)
            config.workers = int((config.expt.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.expt.gpu],
                find_unused_parameters=True
                )
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
    
    # define loss function (criterion) and optimizer
    criterion_pt = nn.CosineSimilarity(dim=1).cuda(config.expt.gpu)
    criterion_ft = nn.CrossEntropyLoss().cuda(config.expt.gpu)
    
    optim_params_pt = [{
        'params': model.module.backbone.parameters(),
        'fix_lr': False
        },
        {
            'params': model.module.encoder_head.parameters(),
            'fix_lr': False
            },
        {
            'params': model.module.predictor.parameters(),
            'fix_lr': config.simsiam.fix_pred_lr
            }]
    
    print(f"world size: {torch.distributed.get_world_size()}")
    print(f"finetuning bs: {config.finetuning.batch_size}")
    print(f"finetuning lr: {config.finetuning.lr}")
    print(f"init_lr_ft: {init_lr_ft}")
    
    print(f"pre-training bs: {config.train.batch_size}")
    print(f"pre-training lr: {config.train.lr}")
    print(f"init_lr_pt: {init_lr_pt}")
    
    optimizer_pt = torch.optim.SGD(
        optim_params_pt, init_lr_pt,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay
        )
    
    optimizer_ft = torch.optim.SGD(
        model.module.classifier_head.parameters(), init_lr_ft,
        momentum=config.finetuning.momentum,
        weight_decay=config.finetuning.weight_decay
        )
    
    # in case a dumped model exist and ssl_model_checkpoint is not set, load that dumped model
    newest_model = get_newest_model(expt_dir)
    if newest_model and config.expt.ssl_model_checkpoint_path is None:
        config.expt.ssl_model_checkpoint_path = newest_model
    
    total_iter = 0
    
    # optionally resume from a checkpoint
    if config.expt.ssl_model_checkpoint_path:
        if os.path.isfile(config.expt.ssl_model_checkpoint_path):
            print(f"=> loading checkpoint '{config.expt.ssl_model_checkpoint_path}'")
            if config.expt.gpu is None:
                checkpoint = torch.load(config.expt.ssl_model_checkpoint_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'cuda:{config.expt.gpu}'
                checkpoint = torch.load(config.expt.ssl_model_checkpoint_path, map_location=loc)
            config.train.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_pt.load_state_dict(checkpoint['optimizer_pt'])
            optimizer_ft.load_state_dict(checkpoint['optimizer_ft'])
            total_iter = checkpoint['total_iter']
            print(f"=> loaded checkpoint '{config.expt.ssl_model_checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{config.expt.ssl_model_checkpoint_path}'")
    
    # Data loading code
    traindir = os.path.join(config.data.dataset, 'train')
    
    train_loader_pt, train_sampler_pt, train_loader_ft, train_sampler_ft, test_loader_ft = get_loaders(traindir, config)
    
    cudnn.benchmark = True
    writer = None
    
    if config.expt.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(expt_dir, "tensorboard"))
    
    for epoch in range(config.train.start_epoch, config.train.epochs):
        
        if config.expt.distributed:
            train_sampler_pt.set_epoch(epoch)
            train_sampler_ft.set_epoch(epoch)
        
        # train for one epoch
        cur_lr_pt = adjust_learning_rate(optimizer_pt, init_lr_pt, epoch, config.train.epochs)
        cur_lr_ft = adjust_learning_rate(optimizer_ft, init_lr_ft, epoch, config.finetuning.epochs)
        print(f"current pretrain lr: {cur_lr_pt}, finetune lr: {cur_lr_ft}")
        
        warmup = config.train.warmup > epoch
        print(f"Warmup status: {warmup}")
        
        total_iter = train_one_epoch(
            train_loader_pt=train_loader_pt,
            train_loader_ft=train_loader_ft,
            model=model,
            criterion_pt=criterion_pt,
            criterion_ft=criterion_ft,
            optimizer_pt=optimizer_pt,
            optimizer_ft=optimizer_ft,
            epoch=epoch,
            total_iter=total_iter,
            config=config,
            writer=writer,
            advanced_stats=config.expt.advanced_stats,
            warmup=warmup,
            layer_wise_stats=config.expt.layer_wise_stats
            )
        
        # evaluate on validation set
        if epoch % config.expt.eval_freq == 0:
            top1_avg = validate(test_loader_ft, model, criterion_ft, config)
            if config.expt.rank == 0:
                writer.add_scalar('Test/Accuracy@1', top1_avg, total_iter)
        
        check_and_save_checkpoint(config, ngpus_per_node, total_iter, epoch, model, optimizer_pt, optimizer_ft, expt_dir)
    
    if config.expt.rank == 0:
        writer.close()


def train_one_epoch(
    train_loader_pt,
    train_loader_ft,
    model,
    criterion_pt,
    criterion_ft,
    optimizer_pt,
    optimizer_ft,
    epoch,
    total_iter,
    config,
    writer,
    advanced_stats=False,
    warmup=False,
    layer_wise_stats=False
    ):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_pt = AverageMeter('Loss PT', ':.4f')
    losses_ft = AverageMeter('Loss FT', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    if advanced_stats:
        if layer_wise_stats:
            cos_sim_avg = AverageMeter('Cos. Sim. PT-FT layer-wise average', ':6.4f')
            cos_sim_std = AverageMeter('Cos. Sim. PT-FT layer-wise std.', ':6.4f')
            dot_prod_avg = AverageMeter('Dot Product PT-FT layer-wise average', ':6.4f')
            dot_prod_std = AverageMeter('Dot Product PT-FT layer-wise std.', ':6.4f')
            eucl_dis_avg = AverageMeter('Eucl. Dist. PT-FT layer-wise average', ':6.4f')
            eucl_dis_std = AverageMeter('Eucl. Dist. PT-FT layer-wise std.', ':6.4f')
            norm_pt_avg = AverageMeter('Norm PT layer-wise average', ':6.4f')
            norm_pt_std = AverageMeter('Norm PT layer-wise std.', ':6.4f')
            norm_ft_avg = AverageMeter('Norm FT layer-wise average', ':6.4f')
            norm_ft_std = AverageMeter('Norm FT layer-wise std.', ':6.4f')
            meters = [batch_time, losses_pt, losses_ft, top1, cos_sim_avg, cos_sim_std, dot_prod_avg, dot_prod_std, eucl_dis_avg, eucl_dis_std, norm_pt_avg, norm_pt_std, norm_ft_avg, norm_ft_std]
        else:
            cos_sim = AverageMeter('Cos. Sim. PT-FT', ':6.4f')
            dot_prod = AverageMeter('Dot Product PT-FT', ':6.4f')
            eucl_dis = AverageMeter('Eucl. Dist. PT-FT', ':6.4f')
            norm_pt = AverageMeter('Norm PT', ':6.4f')
            norm_ft = AverageMeter('Norm FT', ':6.4f')
            # removed data_time and top5 due to brevity
            meters = [batch_time, losses_pt, losses_ft, top1, cos_sim, dot_prod, eucl_dis, norm_pt, norm_ft]
    else:
        meters = [batch_time, data_time, losses_pt, losses_ft, top1, top5]
    
    progress = ProgressMeter(
        num_batches=len(train_loader_pt),
        meters=meters,
        prefix=f"Epoch: [{epoch}]"
        )
    
    end = time.time()
    assert len(train_loader_pt) <= len(train_loader_ft), 'So since this seems to break, we should write code to run multiple finetune epoch per pretrain epoch'
    for i, ((images_pt, _), (images_ft, target_ft)) in enumerate(zip(train_loader_pt, train_loader_ft)):
        
        total_iter += 1
        advanced_stats_meters = []
        
        if config.expt.gpu is not None:
            images_pt[0] = images_pt[0].cuda(config.expt.gpu, non_blocking=True)
            images_pt[1] = images_pt[1].cuda(config.expt.gpu, non_blocking=True)
            images_ft = images_ft.cuda(config.expt.gpu, non_blocking=True)
            target_ft = target_ft.cuda(config.expt.gpu, non_blocking=True)
        
        loss_pt, backbone_grads_pt = pretrain(model, images_pt, criterion_pt, optimizer_pt, losses_pt, data_time, end, advanced_stats=advanced_stats, layer_wise_stats=layer_wise_stats)
        
        if not warmup:
            loss_ft, backbone_grads_ft = finetune(model, images_ft, target_ft, criterion_ft, optimizer_ft, losses_ft, top1, top5, advanced_stats=advanced_stats, layer_wise_stats=layer_wise_stats)
        else:
            losses_ft.update(np.inf)
            loss_ft = np.inf
        
        if advanced_stats:
            if not warmup:
                if layer_wise_stats:
                    mean, std = calc_layer_wise_stats(F.cosine_similarity, backbone_grads_pt, backbone_grads_ft, metric_type="cosine")
                    cos_sim_avg.update(mean), cos_sim_std.update(std)
                    
                    mean, std = calc_layer_wise_stats(torch.dot, backbone_grads_pt, backbone_grads_ft, metric_type="dot")
                    dot_prod_avg.update(mean), dot_prod_std.update(std)
                    
                    mean, std = calc_layer_wise_stats(torch.linalg.norm, backbone_grads_pt, backbone_grads_ft, metric_type="euclidean")
                    eucl_dis_avg.update(mean), eucl_dis_std.update(std)
                    
                    mean, std = calc_layer_wise_stats(torch.linalg.norm, backbone_grads_pt, metric_type="norm")
                    norm_pt_avg.update(mean), norm_pt_std.update(std)
                    
                    mean, std = calc_layer_wise_stats(torch.linalg.norm, backbone_grads_ft, metric_type="norm")
                    norm_ft_avg.update(mean), norm_ft_std.update(std)
                
                else:
                    cos_sim.update(F.cosine_similarity(backbone_grads_pt, backbone_grads_ft, dim=0))
                    dot_prod.update(torch.dot(backbone_grads_pt, backbone_grads_ft))
                    eucl_dis.update(torch.linalg.norm(backbone_grads_pt - backbone_grads_ft, 2))
                    norm_pt.update(torch.linalg.norm(backbone_grads_pt, 2))
                    norm_ft.update(torch.linalg.norm(backbone_grads_ft, 2))
            else:
                # no resetting needed, as meters are freshly initialized at each epoch
                if layer_wise_stats:
                    cos_sim_avg.update(0.), cos_sim_std.update(0.)
                    dot_prod_avg.update(0.), dot_prod_std.update(0.)
                    eucl_dis_avg.update(0.), eucl_dis_std.update(0.)
                    
                    mean, std = calc_layer_wise_stats(torch.linalg.norm, backbone_grads_pt)
                    norm_pt_avg.update(mean), norm_pt_std.update(std)
                    
                    norm_ft_avg.update(0.), norm_ft_std.update(0.)
                else:
                    cos_sim.update(0.)
                    dot_prod.update(0.)
                    eucl_dis.update(0.)
                    norm_pt.update(torch.linalg.norm(backbone_grads_pt, 2))
                    norm_ft.update(0.)
        
        if advanced_stats and layer_wise_stats:
            advanced_stats_meters = [cos_sim_avg, cos_sim_std, dot_prod_avg, dot_prod_std, eucl_dis_avg, eucl_dis_std, norm_pt_avg, norm_pt_std, norm_ft_avg, norm_ft_std]
        elif advanced_stats:
            advanced_stats_meters = [cos_sim, dot_prod, eucl_dis, norm_pt, norm_ft]
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % config.expt.print_freq == 0:
            progress.display(i)
        if config.expt.rank == 0:
            write_to_summary_writer(total_iter, loss_pt, loss_ft, data_time, batch_time, optimizer_pt, optimizer_ft, top1, top5, advanced_stats_meters, writer)
    
    return total_iter


def pretrain(model, images_pt, criterion_pt, optimizer_pt, losses_pt, data_time, end, advanced_stats=False, layer_wise_stats=False):
    # backbone_grads = np.empty(sum(p.numel() for p in model.parameters()))
    if layer_wise_stats:
        backbone_grads = OrderedDict()
    else:
        backbone_grads = torch.Tensor().cuda()
    # backbone_grads = OrderedDict()
    
    model.requires_grad_(True)
    
    # switch to train mode
    model.train()
    
    # measure data loading time
    data_time.update(time.time() - end)
    
    # pre-training
    # compute outputs
    p1, p2, z1, z2 = model(x1=images_pt[0], x2=images_pt[1], finetuning=False)
    
    # compute losses
    loss_pt = -(criterion_pt(p1, z2).mean() + criterion_pt(p2, z1).mean()) * 0.5
    losses_pt.update(loss_pt.item(), images_pt[0].size(0))
    
    # compute gradient and do SGD step
    optimizer_pt.zero_grad()
    loss_pt.backward()
    # step does not change .grad field of the parameters.
    optimizer_pt.step()
    
    if advanced_stats:
        for key, param in model.module.backbone.named_parameters():
            if layer_wise_stats:
                backbone_grads[key] = torch.tensor(param.grad.detach().clone().flatten())
            else:
                backbone_grads = torch.cat([backbone_grads, param.grad.detach().clone().flatten()], dim=0)
            # backbone_grads = np.concatenate((backbone_grads, param.grad.detach().clone().flatten().cpu()))
            # backbone_grads[key] = param.grad.detach().clone().flatten().cpu()
    
    return loss_pt, backbone_grads


def finetune(model, images_ft, target_ft, criterion_ft, optimizer_ft, losses_ft, top1, top5, advanced_stats=False, layer_wise_stats=False):
    if layer_wise_stats:
        backbone_grads = OrderedDict()
    else:
        backbone_grads = torch.Tensor().cuda()
    # backbone_grads = OrderedDict()
    
    # fine-tuning
    model.eval()
    
    optimizer_ft.zero_grad()
    # in finetuning mode, we only optimize the classifier head's parameters
    # -> turn on backbone params grad computation before forward is called
    if advanced_stats:
        model.module.backbone.requires_grad_(True)
    else:
        model.module.backbone.requires_grad_(False)
    
    model.module.classifier_head.requires_grad_(True)
    
    # compute outputs
    output_ft = model(images_ft, finetuning=True)
    loss_ft = criterion_ft(output_ft, target_ft)
    loss_ft.backward()
    
    if advanced_stats:
        for key, param in model.module.backbone.named_parameters():
            if layer_wise_stats:
                backbone_grads[key] = torch.tensor(param.grad.detach().clone().flatten())
            else:
                backbone_grads = torch.cat([backbone_grads, param.grad.detach().clone().flatten()], dim=0)
            # backbone_grads = np.concatenate((backbone_grads, param.grad.detach().clone().flatten().cpu()))
            # backbone_grads[key] = param.grad.detach().clone().flatten().cpu()
    
    # compute losses and measure accuracy
    acc1, acc5 = accuracy(output_ft, target_ft, topk=(1, 5))
    losses_ft.update(loss_ft.item(), images_ft.size(0))
    top1.update(acc1[0], images_ft.size(0))
    top5.update(acc5[0], images_ft.size(0))
    
    # only optimizes classifier head parameters
    optimizer_ft.step()
    
    # just to make sure to prevent grad leakage
    for param in model.module.parameters():
        param.grad = None
    
    return loss_ft, backbone_grads


def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
            return init_lr
        else:
            param_group['lr'] = cur_lr
            return cur_lr


def get_loaders(traindir, config):
    train_loader_pt, _, train_sampler_pt, _ = get_train_valid_loader(
        data_dir=traindir,
        batch_size=config.train.batch_size,
        random_seed=config.expt.seed,
        valid_size=0.0,
        dataset_name=config.data.dataset,
        shuffle=True,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=False,
        distributed=config.expt.distributed,
        drop_last=True,
        get_fine_tuning_loaders=False,
        )
    
    train_loader_ft, _, train_sampler_ft, _ = get_train_valid_loader(
        data_dir=traindir,
        batch_size=config.finetuning.batch_size,
        random_seed=config.expt.seed,
        valid_size=0.0,
        dataset_name="ImageNet",
        shuffle=True,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=False,
        distributed=config.expt.distributed,
        drop_last=True,
        get_fine_tuning_loaders=True,
        )
    
    test_loader_ft = get_test_loader(
        data_dir=traindir,
        batch_size=256,
        dataset_name="ImageNet",
        shuffle=False,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=False,
        drop_last=False,
        )
    
    return train_loader_pt, train_sampler_pt, train_loader_ft, train_sampler_ft, test_loader_ft


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion, config):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: '
        )
    
    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.expt.gpu is not None:
                images = images.cuda(config.expt.gpu, non_blocking=True)
                target = target.cuda(config.expt.gpu, non_blocking=True)
            
            # compute output
            output = model(images, finetuning=True)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % config.expt.print_freq == 0:
                progress.display(i)
        
        # TODO: this should also be done with the ProgressMeter
        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    return top1.avg


def write_to_summary_writer(total_iter, loss_pt, loss_ft, data_time, batch_time, optimizer_pt, optimizer_ft, top1, top5, advanced_stats_meters, writer):
    writer.add_scalar('Loss/pre-training', loss_pt.item(), total_iter)
    if isinstance(loss_ft, float):
        writer.add_scalar('Loss/fine-tuning', loss_ft, total_iter)
    else:
        writer.add_scalar('Loss/fine-tuning', loss_ft.item(), total_iter)
    writer.add_scalar('Accuracy/@1', top1.val, total_iter)
    writer.add_scalar('Accuracy/@5', top5.val, total_iter)
    writer.add_scalar('Accuracy/@1 average', top1.avg, total_iter)
    writer.add_scalar('Accuracy/@5 average', top5.avg, total_iter)
    writer.add_scalar('Time/Data', data_time.val, total_iter)
    writer.add_scalar('Time/Batch', batch_time.val, total_iter)
    # assuming only one param group
    writer.add_scalar('Learning rate/pre-training', optimizer_pt.param_groups[0]['lr'], total_iter)
    writer.add_scalar('Learning rate/fine-tuning', optimizer_ft.param_groups[0]['lr'], total_iter)

    for stat in advanced_stats_meters:
        if isinstance(stat, ExponentialMovingAverageMeter):
            writer.add_scalar(f'Advanced Stats/{stat.name}', stat.val, total_iter)
            writer.add_scalar(f'Advanced Stats/{stat.name} average', stat.avg, total_iter)
            # exponential moving average
            writer.add_scalar(f'Advanced Stats/{stat.name} exp. moving average', stat.ema, total_iter)
        else:
            writer.add_scalar(f'Advanced Stats/{stat.name}', stat.val, total_iter)
            writer.add_scalar(f'Advanced Stats/{stat.name} average', stat.avg, total_iter)


def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    
    return args


def calc_layer_wise_stats(metric, backbone_grads_pt, backbone_grads_ft=None, metric_type="cosine"):
    allowed_types = ["cosine", "euclidean", "norm", "dot"]
    assert metric_type in allowed_types, f"metric_type must be one of {allowed_types}"
    metric_vals = []
    if backbone_grads_ft:
        for (k1, v1), (k2, v2) in zip(backbone_grads_pt.items(), backbone_grads_ft.items()):
            if k1 == k2 and "bn" not in k1:
                if metric_type == "euclidean":
                    metric_vals.append(metric(v1 - v2, 2).cpu().numpy())
                elif metric_type == "dot":
                    metric_vals.append(metric(v1, v2).cpu().numpy())
                elif metric_type == "cosine":
                    metric_vals.append(metric(v1, v2, dim=0).cpu().numpy())
    else:
        for k1, v1 in backbone_grads_pt.items():
            if metric_type == "norm":
                metric_vals.append(metric(v1, 2).cpu().numpy())

    return np.mean(metric_vals), np.std(metric_vals)


if __name__ == '__main__':
    user = os.environ.get('USER')
    
    config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config file path.')
    config_parser.add_argument('--config', default="metassl/default_metassl_config.yaml", help='Select which yaml file to use depending on the selected experiment mode')
    parser = ArgumentParser()
    
    parser.add_argument('--expt', default="expt", type=str, metavar='N')
    parser.add_argument('--expt.expt_name', default='pre-training-fix-lr-100-256', type=str, help='experiment name')
    parser.add_argument('--expt.expt_mode', default='ImageNet', choices=["ImageNet", "CIFAR10"], help='Define which dataset to use to select the correct yaml file.')
    parser.add_argument('--expt.save_model', action='store_false', help='save the model to disc or not (default: True)')
    parser.add_argument('--expt.save_model_frequency', default=1, type=int, metavar='N', help='save model frequency in # of epochs')
    parser.add_argument('--expt.ssl_model_checkpoint_path', type=str, help='ppath to the pre-trained model, resumes training if model with same config exists')
    parser.add_argument('--expt.target_model_checkpoint_path', type=str, help='path to the downstream task model, resumes training if model with same config exists')
    parser.add_argument('--expt.print_freq', default=10, type=int, metavar='N')
    parser.add_argument('--expt.gpu', default=None, type=int, metavar='N', help='GPU ID to train on (if not distributed)')
    parser.add_argument('--expt.multiprocessing_distributed', action='store_false', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training (default: True)')
    parser.add_argument('--expt.dist_backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--expt.dist_url', type=str, default='tcp://localhost:10005', help='url used to set up distributed training')
    parser.add_argument('--expt.workers', default=32, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--expt.rank', default=0, type=int, metavar='N', help='node rank for distributed training')
    parser.add_argument('--expt.world_size', default=1, type=int, metavar='N', help='number of nodes for distributed training')
    parser.add_argument('--expt.eval_freq', default=10, type=int, metavar='N', help='every eval_freq epoch will the model be evaluated')
    parser.add_argument('--expt.seed', default=123, type=int, metavar='N', help='random seed of numpy and torch')
    parser.add_argument('--expt.evaluate', action='store_true', help='evaluate model on validation set once and terminate (default: False)')
    parser.add_argument('--expt.advanced_stats', action='store_false', help='compute advanced stats such as cosine similarity and dot product, only used in alternating mode (default: True)')
    parser.add_argument('--expt.layer_wise_stats', action='store_true', help='compute the advanced stats for each layer separately and then plot the average and deviation.')
    
    parser.add_argument('--train', default="train", type=str, metavar='N')
    parser.add_argument('--train.batch_size', default=256, type=int, metavar='N', help='in distributed setting this is the total batch size, i.e. batch size = individual bs * number of GPUs')
    parser.add_argument('--train.epochs', default=100, type=int, metavar='N', help='number of pre-training epochs')
    parser.add_argument('--train.start_epoch', default=0, type=int, metavar='N', help='start training at epoch n')
    parser.add_argument('--train.optimizer', type=str, default='sgd', help='optimizer type, options: sgd')
    parser.add_argument('--train.schedule', type=str, default='cosine', help='learning rate schedule, not implemented')
    parser.add_argument('--train.warmup', default=0, type=int, metavar='N', help='denotes the number of epochs that we only pre-train without finetuning afterwards; warmup is turned off when set to 0')
    parser.add_argument('--train.weight_decay', default=0.0001, type=float, metavar='N')
    parser.add_argument('--train.momentum', default=0.9, type=float, metavar='N', help='SGD momentum')
    parser.add_argument('--train.lr', default=0.05, type=float, metavar='N', help='pre-training learning rate')
    
    parser.add_argument('--finetuning', default="finetuning", type=str, metavar='N')
    parser.add_argument('--finetuning.batch_size', default=256, type=int, metavar='N', help='in distributed setting this is the total batch size, i.e. batch size = individual bs * number of GPUs')
    parser.add_argument('--finetuning.epochs', default=100, type=int, metavar='N', help='number of pre-training epochs')
    parser.add_argument('--finetuning.start_epoch', default=0, type=int, metavar='N', help='start training at epoch n')
    parser.add_argument('--finetuning.optimizer', type=str, default='sgd', help='optimizer type, options: sgd')
    parser.add_argument('--finetuning.schedule', type=str, default='cosine', help='learning rate schedule, not implemented')
    parser.add_argument('--finetuning.weight_decay', default=0.0, type=float, metavar='N')
    parser.add_argument('--finetuning.momentum', default=0.9, type=float, metavar='N', help='SGD momentum')
    parser.add_argument('--finetuning.lr', default=100, type=float, metavar='N', help='finetuning learning rate')
    
    parser.add_argument('--model', default="model", type=str, metavar='N')
    parser.add_argument('--model.model_type', type=str, default='resnet50', help='all torchvision ResNets')
    parser.add_argument('--model.seed', type=int, default=123, help='the seed')
    parser.add_argument('--model.turn_off_bn', action='store_true', help='turns off all batch norm instances in the model')
    
    parser.add_argument('--data', default="data", type=str, metavar='N')
    parser.add_argument('--data.seed', type=int, default=123, help='the seed')
    parser.add_argument('--data.dataset', type=str, default="ImageNet", help='supported datasets: CIFAR10, CIFAR100, ImageNet')
    parser.add_argument('--data.data_dir', type=str, default=f"/home/{user}/workspace/data/metassl", help='supported datasets: CIFAR10, CIFAR100, ImageNet')
    
    parser.add_argument('--simsiam', default="simsiam", type=str, metavar='N')
    parser.add_argument('--simsiam.dim', type=int, default=2048, help='the feature dimension')
    parser.add_argument('--simsiam.pred_dim', type=int, default=512, help='the hidden dimension of the predictor')
    parser.add_argument('--simsiam.fix_pred_lr', action="store_false", help='fix learning rate for the predictor (default: True')
    
    args = _parse_args(config_parser, parser)
    
    # Saving checkpoint and config pased on experiment mode
    if args.expt.expt_mode == "ImageNet":
        expt_dir = f"/home/{user}/workspace/experiments/metassl"
    elif args.expt.expt_mode == "CIFAR10":
        expt_dir = "experiments"
    else:
        raise ValueError(f"Experiment mode {args.expt.expt_mode} is undefined!")
    expt_sub_dir = os.path.join(expt_dir, args.expt.expt_name)
    
    expt_dir = pathlib.Path(expt_dir)
    
    if not os.path.exists(expt_sub_dir):
        os.makedirs(expt_sub_dir)
    
    with open(os.path.join(expt_sub_dir, "config.yaml"), "w") as f:
        yaml.dump(args, f)
        print(f"copied config to {f.name}")
    
    config = AttrDict(jsonargparse.namespace_to_dict(args))
    
    main(config=config, expt_dir=expt_sub_dir)
