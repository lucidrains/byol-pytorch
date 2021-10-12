# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# code taken from https://github.com/facebookresearch/simsiam

import argparse
import builtins
import math
import os
import pathlib
import random
import shutil
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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

try:
    # For execution in PyCharm
    from metassl.utils.data import get_train_valid_loader, get_test_loader
    from metassl.utils.config import AttrDict
    from metassl.utils.meters import AverageMeter, ProgressMeter
    from metassl.utils.simsiam_alternating import SimSiam
    import metassl.models.resnet_cifar as our_cifar_resnets
except ImportError:
    # For execution in command line
    from .utils.data import get_train_valid_loader, get_test_loader
    from .utils.config import AttrDict
    from .utils.meters import AverageMeter, ProgressMeter
    from .utils.simsiam_alternating import SimSiam
    from .models import resnet_cifar as our_cifar_resnets

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
    
    # infer learning rate before changing batch size
    
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
    # print(model) # print model after SyncBatchNorm
    
    # define loss function (criterion) and optimizer
    pt_criterion = nn.CosineSimilarity(dim=1).cuda(config.expt.gpu)
    ft_criterion = nn.CrossEntropyLoss().cuda(config.expt.gpu)
    
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
    
    init_lr_pt = config.train.lr * config.train.batch_size / 256
    init_lr_ft = config.finetuning.lr * config.finetuning.batch_size / 256
    print(f"finetuning bs: {config.finetuning.batch_size}")
    print(f"finetuning lr: {config.finetuning.lr }")
    print(f"init_lr_ft: {init_lr_ft}")
    
    
    pt_optimizer = torch.optim.SGD(
        optim_params_pt, init_lr_pt,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay
        )
    
    ft_optimizer = torch.optim.SGD(
        model.module.classifier_head.parameters(), init_lr_ft,
        momentum=config.finetuning.momentum,
        weight_decay=config.finetuning.weight_decay
        )
    
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
            pt_optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{config.expt.ssl_model_checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{config.expt.ssl_model_checkpoint_path}'")
    
    # Data loading code
    traindir = os.path.join(config.data.dataset, 'train')
    
    pt_train_loader, pt_train_sampler, ft_train_loader, ft_train_sampler, ft_test_loader = get_loaders(traindir, config)
    
    cudnn.benchmark = True
    
    for epoch in range(config.train.start_epoch, config.train.epochs):
        if config.expt.distributed:
            pt_train_sampler.set_epoch(epoch)
            ft_train_sampler.set_epoch(epoch)
            
        # train for one epoch
        print(f"preparing pretraining at epoch {epoch}")
        cur_lr_pt = adjust_learning_rate(pt_optimizer, init_lr_pt, epoch, config.train.epochs)
        cur_lr_ft = adjust_learning_rate(ft_optimizer, init_lr_ft, epoch, config.finetuning.epochs)
        print(f"current pretrain lr: {cur_lr_pt}, finetune lr: {cur_lr_ft}")
        
        train_one_epoch(pt_train_loader, ft_train_loader, model, pt_criterion, ft_criterion, pt_optimizer, ft_optimizer, epoch, config)
        
        # evaluate on validation set
        validate(ft_test_loader, model, ft_criterion, config)
        
        if not config.expt.multiprocessing_distributed or (config.expt.multiprocessing_distributed
                                                           and config.expt.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch':      epoch + 1,
                    'arch':       config.model.model_type,
                    'state_dict': model.state_dict(),
                    'optimizer':  pt_optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(expt_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                )
            # if epoch == config.train.start_epoch and model_prev_state_dct:
            #     sanity_check(model.state_dict(), model_prev_state_dct)


def train_one_epoch(train_loader_pt, train_loader_ft, model, criterion_pt, criterion_ft, optimizer_pt, optimizer_ft, epoch, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_pt = AverageMeter('Loss PT', ':.4f')
    losses_ft = AverageMeter('Loss FT', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader_pt),
        [batch_time, data_time, losses_pt, losses_ft, top1, top5],
        prefix=f"Epoch: [{epoch}]"
        )
    
    # switch to train mode
    model.train()
    
    end = time.time()
    assert len(train_loader_pt) <= len(train_loader_ft), \
        'So since this seems to break, we should write code to run multiple finetune epoch per pretrain epoch'
    for i, ((pt_images, _), (ft_images, ft_target)) in enumerate(zip(train_loader_pt, train_loader_ft)):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        if config.expt.gpu is not None:
            pt_images[0] = pt_images[0].cuda(config.expt.gpu, non_blocking=True)
            pt_images[1] = pt_images[1].cuda(config.expt.gpu, non_blocking=True)
            ft_images = ft_images.cuda(config.expt.gpu, non_blocking=True)
        
        ft_target = ft_target.cuda(config.expt.gpu, non_blocking=True)
        
        # pre-training
        # compute outputs
        p1, p2, z1, z2 = model(x1=pt_images[0], x2=pt_images[1])
        
        # compute losses
        loss_pt = -(criterion_pt(p1, z2).mean() + criterion_pt(p2, z1).mean()) * 0.5
        losses_pt.update(loss_pt.item(), pt_images[0].size(0))
        
        # compute gradient and do SGD step
        optimizer_pt.zero_grad()
        loss_pt.backward()
        optimizer_pt.step()
        
        # fine-tuning
        model.eval()
        # compute outputs
        ft_output = model(ft_images, finetuning=True)
        loss_ft = criterion_ft(ft_output, ft_target)
        
        # compute losses and measure accuracy
        acc1, acc5 = accuracy(ft_output, ft_target, topk=(1, 5))
        losses_ft.update(loss_ft.item(), ft_images.size(0))
        top1.update(acc1[0], ft_images.size(0))
        top5.update(acc5[0], ft_images.size(0))
        
        optimizer_ft.zero_grad()
        loss_ft.backward()
        optimizer_ft.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % config.expt.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    pt_train_loader, _, pt_train_sampler, _ = get_train_valid_loader(
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
    
    ft_train_loader, _, ft_train_sampler, _ = get_train_valid_loader(
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
    
    ft_test_loader = get_test_loader(
        data_dir=traindir,
        batch_size=256,
        dataset_name="ImageNet",
        shuffle=False,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=False,
        drop_last=False,
        )
    
    return pt_train_loader, pt_train_sampler, ft_train_loader, ft_train_sampler, ft_test_loader


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


if __name__ == '__main__':
    user = os.environ.get('USER')
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--expt_name', default='pre-training-fix-lr-100-256', type=str, help='experiment name')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument(
        '--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
        help='initial (base) learning rate', dest='lr'
        )
    parser.add_argument('--ssl_model_checkpoint_path', default=None, type=str, help='pretrained model checkpoint path')
    parser.add_argument(
        '--expt_mode', default="ImageNet", choices=["ImageNet", "CIFAR10"],
        help='Define which dataset to use to select the correct yaml file.'
        )
    args = parser.parse_args()
    
    expt_name = args.expt_name
    epochs = args.epochs
    lr = args.lr
    ssl_model_checkpoint_path = args.ssl_model_checkpoint_path
    
    # Saving checkpoint and config pased on experiment mode
    if args.expt_mode == "ImageNet":
        expt_dir = f"/home/{user}/workspace/experiments/metassl"
    elif args.expt_mode == "CIFAR10":
        expt_dir = "experiments"
    else:
        raise ValueError(f"Experiment mode {args.expt_mode} is undefined!")
    expt_sub_dir = os.path.join(expt_dir, expt_name)
    
    expt_dir = pathlib.Path(expt_dir)
    
    if not os.path.exists(expt_sub_dir):
        os.makedirs(expt_sub_dir)
    
    # Select which yaml file to use depending on the selected experiment mode
    if args.expt_mode == "ImageNet":
        config_path = "metassl/default_metassl_config.yaml"
    elif args.expt_mode == "CIFAR10":
        config_path = "metassl/default_metassl_config_cifar10.yaml"
    else:
        raise ValueError(f"Experiment mode {args.expt_mode} is undefined!")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if args.expt_mode == "ImageNet":
        config['data']['data_dir'] = f'/home/{user}/workspace/data/metassl'
        config['expt']['expt_name'] = expt_name
        config['expt']['ssl_model_checkpoint_path'] = ssl_model_checkpoint_path
        config['train']['epochs'] = epochs
        config['train']['lr'] = lr
    
    print(expt_name, ssl_model_checkpoint_path, epochs, lr)
    print(f"batch size {config['train']['batch_size']}")
    
    with open(os.path.join(expt_sub_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        print(f"copied config to {expt_sub_dir}")
    
    config = AttrDict(config)
    
    main(config=config, expt_dir=expt_sub_dir)
