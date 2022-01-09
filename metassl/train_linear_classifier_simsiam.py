#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
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

try:
    # For execution in PyCharm
    from metassl.utils.data import get_train_valid_loader, get_test_loader
    from metassl.utils.config import AttrDict
    from metassl.utils.meters import AverageMeter, ProgressMeter
    from metassl.utils.torch_utils import accuracy
    import metassl.models.resnet_cifar as our_cifar_resnets

except ImportError:
    # For execution in command line
    from .utils.data import get_train_valid_loader, get_test_loader
    from .utils.config import AttrDict
    from .utils.meters import AverageMeter, ProgressMeter
    from .utils.torch_utils import accuracy, validate
    from .models import resnet_cifar as our_cifar_resnets

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)

best_acc1 = 0


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
    global best_acc1
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
            backend=config.expt.dist_backend,
            init_method=config.expt.dist_url,
            world_size=config.expt.world_size,
            rank=config.expt.rank
        )
        torch.distributed.barrier()
    # create model
    print(f"=> creating model '{config.model.model_type}'")

    # TODO: this part below is different
    # create for CIFAR10, arg values taken from simsiam.py
    if config.data.dataset == 'CIFAR10':
        model = our_cifar_resnets.resnet18(num_classes=10, zero_init_residual=True)
    else:
        model = models.__dict__[config.model.model_type]()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained (ssl model), before DistributedDataParallel constructor
    if config.expt.ssl_model_checkpoint_path:
        if os.path.isfile(config.expt.ssl_model_checkpoint_path):
            print(f"=> loading pre-trained model checkpoint '{config.expt.ssl_model_checkpoint_path}'")
            checkpoint = torch.load(config.expt.ssl_model_checkpoint_path, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']

            has_encoder = False
            has_backbone = False

            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                    has_encoder = True

                if k.startswith('module.backbone') and not k.startswith('module.backbone.fc'):
                    # remove prefix
                    state_dict[k[len("module.backbone."):]] = state_dict[k]
                    has_backbone = True

                # delete renamed or unused k
                del state_dict[k]

            assert has_backbone != has_encoder

            config.finetuning.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, f"{msg}"

            print(f"=> loaded pre-trained model '{config.expt.ssl_model_checkpoint_path}'")
        else:
            print(f"=> no checkpoint found at '{config.expt.ssl_model_checkpoint_path}'")

    # infer learning rate before changing batch size
    init_lr = config.finetuning.lr * config.finetuning.batch_size / 256

    if config.expt.distributed:
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
            config.expt.workers = int((config.expt.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.expt.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.expt.gpu is not None:
        torch.cuda.set_device(config.expt.gpu)
        model = model.cuda(config.expt.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if config.model.model_type.startswith('alexnet') or config.model.model_type.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(config.expt.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(
        parameters, init_lr,
        momentum=config.finetuning.momentum,
        weight_decay=config.finetuning.weight_decay
    )
    if config.finetuning.optimizer == "lars":
        print("=> use LARS optimizer.")
        # from torchlars import LARS
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    # optionally resume from a checkpoint (target task model)
    if config.expt.target_model_checkpoint_path:
        if os.path.isfile(config.expt.target_model_checkpoint_path):
            print("=> loading checkpoint '{}'".format(config.expt.target_model_checkpoint_path))
            if config.expt.gpu is None:
                checkpoint = torch.load(config.expt.target_model_checkpoint_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'cuda:{config.expt.gpu}'
                checkpoint = torch.load(config.expt.target_model_checkpoint_path, map_location=loc)
            config.finetuning.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if config.expt.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(config.expt.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{config.expt.target_model_checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{config.expt.target_model_checkpoint_path}'")

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(config.data.dataset, 'train')

    train_loader, _, train_sampler, _ = get_train_valid_loader(
        data_dir=traindir,
        batch_size=config.finetuning.batch_size,
        random_seed=config.expt.seed,
        valid_size=0.0,
        dataset_name=config.data.dataset,
        shuffle=True,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=config.download_data,
        distributed=config.expt.distributed,
        drop_last=False,
        get_fine_tuning_loaders=True,
    )

    test_loader = get_test_loader(
        data_dir=traindir,
        batch_size=256,
        dataset_name=config.data.dataset,
        shuffle=False,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=config.download_data,
        drop_last=False,
    )

    if config.expt.evaluate:
        validate(test_loader, model, criterion, config)
        return

    for epoch in range(config.finetuning.start_epoch, config.finetuning.epochs):
        if config.expt.distributed:
            train_sampler.set_epoch(epoch)
        cur_lr = adjust_learning_rate(optimizer, init_lr, epoch, config.finetuning.epochs, config)
        print(f"Current Learing Rate: {cur_lr}")

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config)

        # evaluate on validation set
        acc1 = validate(test_loader, model, criterion, config)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not config.expt.multiprocessing_distributed or (config.expt.multiprocessing_distributed
                                                           and config.expt.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': config.model.model_type,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, filename=os.path.join(expt_dir, f'lin_class_checkpoint_{epoch:04d}.pth.tar')
            )
            if epoch == config.train.start_epoch:
                sanity_check(model.state_dict(), config.expt.ssl_model_checkpoint_path)


def train(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch)
    )

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if i == 0:
            print(f"batch size per gpu {len(images)}")

        if config.expt.gpu is not None:
            images = images.cuda(config.expt.gpu, non_blocking=True)
        target = target.cuda(config.expt.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.expt.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='lin_class_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs, config):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

    return cur_lr


# OG code from facebook repo as validate func in torch.utils has extra args finetune, used by alternating simsiam model.
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    user = os.environ.get('USER')

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--expt_name', default='pre-training-full-train-data-fix-lr-100-256', type=str,
                        help='experiment name')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial (base) '
                                                                                               'learning rate',
                        dest='lr')
    parser.add_argument(
        '--ssl_model_checkpoint_path',
        default='/home/ferreira/workspace/experiments/metassl/pre-training-full-train-data-fix-lr-100-256'
                '/checkpoint_0099.pth.tar',
        type=str, help='pretrained model checkpoint path'
    )
    parser.add_argument(
        '--target_model_checkpoint_path',
        default='/home/ferreira/workspace/experiments/metassl/finetuning-fix-smaller-lr-0.01-100-256'
                '/lin_class_checkpoint_0063.pth.tar',
        type=str, help='target model checkpoint path'
    )
    parser.add_argument('--expt_mode', default="ImageNet", choices=["ImageNet", "CIFAR10"],
                        help='Define which dataset to use to select the correct yaml file.')
    # use this when training with fewer GPUs
    parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--download_data', action='store_true')  # if needed
    args = parser.parse_args()

    expt_name = args.expt_name
    epochs = args.epochs
    lr = args.lr
    ssl_model_checkpoint_path = args.ssl_model_checkpoint_path
    target_model_checkpoint_path = args.target_model_checkpoint_path

    expt_dir = f"/home/{user}/workspace/experiments/metassl"
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

    print(f"epochs: {epochs}")
    print(f"lr: {lr}")
    print(f"ssl model checkpoint: {ssl_model_checkpoint_path}")
    print(f"target model checkpoint: {target_model_checkpoint_path}")
    print(f"experiment name: {expt_name}")

    config['data']['data_dir'] = f'/home/{user}/workspace/data/metassl'
    config['expt']['expt_name'] = expt_name
    config['expt']['ssl_model_checkpoint_path'] = ssl_model_checkpoint_path
    config['expt']['target_model_checkpoint_path'] = target_model_checkpoint_path
    config['finetuning']['epochs'] = epochs
    config['expt']['workers'] = args.workers
    config['finetuning']['lr'] = lr
    config['download_data'] = args.download_data

    with open(os.path.join(expt_sub_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    config = AttrDict(config)

    main(config=config, expt_dir=expt_sub_dir)
