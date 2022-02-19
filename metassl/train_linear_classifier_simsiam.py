#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import os
import random
import time
import warnings
from collections import OrderedDict
import math

import jsonargparse
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
from jsonargparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", category=UserWarning)

try:
    # For execution in PyCharm
    from metassl.utils.data import get_train_valid_loader, get_test_loader, get_loaders
    from metassl.utils.config import AttrDict, _parse_args
    from metassl.utils.meters import AverageMeter, ProgressMeter, ExponentialMovingAverageMeter, initialize_all_meters, calc_layer_wise_stats
    from metassl.utils.simsiam_alternating import SimSiam
    from metassl.utils.summary import write_to_summary_writer
    import metassl.models.resnet_cifar as our_cifar_resnets
    from metassl.utils.torch_utils import get_newest_model, check_and_save_checkpoint, deactivate_bn, validate, accuracy, adjust_learning_rate
    from metassl.utils.io import get_expt_dir_with_bohb_config_id, organize_experiment_saving, find_free_port

except ImportError:
    # For execution in command line
    from .utils.data import get_train_valid_loader, get_test_loader, get_loaders
    from .utils.config import AttrDict, _parse_args
    from .utils.meters import AverageMeter, ProgressMeter, ExponentialMovingAverageMeter, initialize_all_meters, calc_layer_wise_stats
    from .utils.simsiam_alternating import SimSiam
    from .utils.summary import write_to_summary_writer
    from .models import resnet_cifar as our_cifar_resnets
    from .utils.torch_utils import get_newest_model, check_and_save_checkpoint, deactivate_bn, validate, accuracy, adjust_learning_rate
    from .utils.io import get_expt_dir_with_bohb_config_id, organize_experiment_saving, find_free_port

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
    )


def main(config, expt_dir, bohb_infos=None):
    # BOHB only --------------------------------------------------------------------------------------------------------
    if bohb_infos is not None:
        # Integrate budget based on budget_mode
        if config.bohb.budget_mode == "epochs":
            # TODO: @Diane - Check out how to handle the case where #epochs_pt != #epochs_ft
            config.train.epochs = int(bohb_infos['bohb_budget'])
            config.finetuning.epochs = int(bohb_infos['bohb_budget'])
        else:
            raise ValueError(f"Budget mode '{config.bohb.budget_mode}' not implemented yet!")
        
        # Add --bohb.configspace_mode to bohb_infos
        bohb_infos['bohb_configspace'] = config.bohb.configspace_mode
        
        # Create subfoler for each config_id (directory where tensorboard and checkpoints are being saved)
        expt_dir_id = get_expt_dir_with_bohb_config_id(expt_dir, bohb_infos['bohb_config_id'])
        expt_dir = expt_dir_id
        
        # Define master port (for preventing 'Address already in use error' when more than 1 submitted worker on 1 node)
        # TODO: @Diane - Think for another strategy to handle the problem
        # str_config_id = "".join(str(sub_id) for sub_id in bohb_infos['bohb_config_id'])
        # master_port = str(int(bohb_infos['bohb_budget'])) + str_config_id
        # print(f"{master_port=}")
        # if len(master_port) < 5:
        #     master_port = master_port + str(0)
        # print(f"{master_port=}")
        # config.expt.dist_url = "tcp://localhost:" + master_port
        # print(f"{config.expt.dist_url=}")
        
        print(f"\n\n\n\n\n\n{bohb_infos=}\n\n\n\n\n\n")
    # ------------------------------------------------------------------------------------------------------------------

    if config.data.dataset == "CIFAR10":
        # Define master port (for preventing 'Address already in use error' when submitting more than 1 jobs on 1 node)
        # Code from: https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        master_port = find_free_port()
        config.expt.dist_url = "tcp://localhost:" + str(master_port)
        # if this should still fail: do it via filesystem initialization
        # https://pytorch.org/docs/stable/distributed.html#shared-file-system-initialization

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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, expt_dir, bohb_infos))
    else:
        # Simply call main_worker function
        main_worker(config.expt.gpu, ngpus_per_node, config, expt_dir, bohb_infos)
    
    # BOHB only --------------------------------------------------------------------------------------------------------
    # Read validation metric from the .txt (as for mp.spawn returning values is not trivial)
    if bohb_infos is not None:
        with open(expt_dir + "/current_val_metric.txt", 'r') as f:
            val_metric = f.read()
        print(f"{val_metric=}")
        return float(val_metric)
    # ------------------------------------------------------------------------------------------------------------------


def main_worker(gpu, ngpus_per_node, config, expt_dir, bohb_infos):
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
    if config.data.dataset == 'CIFAR10':
        # Use model from our model folder instead from torchvision!
        print(f"=> creating model resnet18")
        model = SimSiam(our_cifar_resnets.resnet18, config.simsiam.dim, config.simsiam.pred_dim)
    else:
        print(f"=> creating model '{config.model.model_type}'")
        model = SimSiam(models.__dict__[config.model.model_type], config.simsiam.dim, config.simsiam.pred_dim)
    
    if config.model.turn_off_bn:
        print("Turning off BatchNorm in entire model.")
        deactivate_bn(model)
        model.encoder_head[6].bias.requires_grad = True
    
    # infer learning rate before changing batch size
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
            config.expt.workers = int((config.expt.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.expt.gpu],
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
    criterion_ft = nn.CrossEntropyLoss().cuda(config.expt.gpu)

    optimizer_ft = torch.optim.SGD(
        params=model.module.classifier_head.parameters(),
        lr=init_lr_ft,
        momentum=config.finetuning.momentum,
        weight_decay=config.finetuning.weight_decay
        )
    
    # in case a dumped model exist and ssl_model_checkpoint is not set, load that dumped model
    newest_model = get_newest_model(expt_dir, suffix="linear_cls*.pth.tar")
    if not newest_model:
        # if lin class model doesn't exist, get newest pre-training model
        newest_model = get_newest_model(expt_dir)
    if newest_model and config.expt.ssl_model_checkpoint_path is None:
        config.expt.ssl_model_checkpoint_path = newest_model

    total_iter = 0
    meters = None
    
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
            
            if "meters_ft" in checkpoint and checkpoint["meters_ft"]:
                meters = checkpoint["meters_ft"]
            if "epoch_ft" in checkpoint:
                config.finetuning.start_epoch = checkpoint['epoch_ft']
                print(f"=> loaded checkpoint '{config.expt.ssl_model_checkpoint_path}' (ft epoch {checkpoint['epoch_ft']})")
            if "optimizer_ft" in checkpoint and checkpoint['optimizer_ft']:
                optimizer_ft.load_state_dict(checkpoint['optimizer_ft'])
                
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"=> no checkpoint found at '{config.expt.ssl_model_checkpoint_path}'")

    if config.finetuning.valid_size > 0:
        train_loader_pt, train_sampler_pt, train_loader_ft, train_sampler_ft, valid_loader_ft, test_loader_ft = get_loaders(config, parameterize_augmentation=False, bohb_infos=bohb_infos)
    else:  # TODO: @Diane - Checkout and test on *parameterized_aug*
        train_loader_pt, train_sampler_pt, train_loader_ft, train_sampler_ft, test_loader_ft = get_loaders(config, parameterize_augmentation=False, bohb_infos=bohb_infos)

    cudnn.benchmark = True
    writer = None
    
    if config.expt.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(expt_dir, "tensorboard_ft"))
    if not meters:
        meters = initialize_all_meters()
    
    for epoch in range(config.finetuning.start_epoch, config.finetuning.epochs):
        
        if config.expt.distributed:
            train_sampler_ft.set_epoch(epoch)

        cur_lr_ft = adjust_learning_rate(optimizer_ft, init_lr_ft, epoch, total_epochs=config.finetuning.epochs)

        if config.expt.wd_decay_ft:
            # Do annealing
            if epoch == 1:
                for group in optimizer_ft.param_groups:
                    group['weight_decay'] = config.finetuning.wd_start
                    current_weight_decay = group['weight_decay']
            else:
                for group in optimizer_ft.param_groups:
                    group['weight_decay'] = config.finetuning.wd_end + 1 / 2 * (config.finetuning.wd_start - config.finetuning.wd_end) * (1 + math.cos(epoch / config.finetuning.epochs * math.pi))
                    current_weight_decay = group['weight_decay']
            print(f"current weight decay (ft): {current_weight_decay}")

        print(f"current finetune lr: {cur_lr_ft}")

        total_iter = train_one_epoch(
            train_loader_ft=train_loader_ft,
            model=model,
            criterion_ft=criterion_ft,
            optimizer_ft=optimizer_ft,
            epoch=epoch,
            total_iter=total_iter,
            config=config,
            writer=writer,
            meters=meters,
            )

        # Determine wheter to evaluate on the validation or test set
        if config.finetuning.valid_size > 0:
            loader_ft = valid_loader_ft
            writer_scalar_mode = "Valid"
        else:
            loader_ft = test_loader_ft
            writer_scalar_mode = "Test"
        # BOHB only ----------------------------------------------------------------------------------------------------
        # TODO: @Diane - Refactor - no priority
        if bohb_infos is not None and config.bohb.budget_mode == "epochs":
            if epoch % config.expt.eval_freq == 0 or epoch % int(bohb_infos['bohb_budget'] - 1) == 0:
                top1_avg = validate(loader_ft, model, criterion_ft, config, finetuning=True)
                if config.expt.rank == 0:
                    writer.add_scalar(writer_scalar_mode + '/Accuracy@1', top1_avg, total_iter)
        # --------------------------------------------------------------------------------------------------------------
        else:
            # evaluate on validation/test set
            if epoch % config.expt.eval_freq == 0 or epoch % (config.finetuning.epochs - 1) == 0:
                top1_avg = validate(loader_ft, model, criterion_ft, config, finetuning=True)
                if config.expt.rank == 0:
                    writer.add_scalar(writer_scalar_mode + '/Accuracy@1', top1_avg, total_iter)

        # make sure to always save at the end of training
        is_last_epoch = epoch + 1 >= config.finetuning.epochs
        if bohb_infos is not None:
            if config.bohb.budget_mode == "epochs" and epoch % int(bohb_infos['bohb_budget'] - 1) == 0:
                check_and_save_checkpoint(
                    config=config,
                    ngpus_per_node=ngpus_per_node,
                    total_iter=total_iter,
                    epoch=epoch,
                    model=model,
                    optimizer_pt=None,
                    optimizer_ft=optimizer_ft,
                    expt_dir=expt_dir,
                    meters=meters,
                    checkpoint_name="linear_cls",
                )
            elif config.bohb.budget_mode != "epochs":
                raise ValueError("Not implemented yet!")

        elif (config.expt.save_model and epoch % config.expt.save_model_frequency == 0) or (config.expt.save_model and is_last_epoch):
            check_and_save_checkpoint(
                config=config,
                ngpus_per_node=ngpus_per_node,
                total_iter=total_iter,
                epoch=epoch,
                model=model,
                optimizer_pt=None,
                optimizer_ft=optimizer_ft,
                expt_dir=expt_dir,
                meters=meters,
                checkpoint_name="linear_cls",
                )
    
    if config.expt.rank == 0:
        writer.close()
        
        batch_time_meter = meters["batch_time_meter"]
        print(f"total batch time elapsed: {batch_time_meter.sum:.2f}s")
    
    # BOHB only --------------------------------------------------------------------------------------------------------
    # Save validation metric in a .txt (as for mp.spawn returning values is not trivial)
    if bohb_infos is not None:
        with open(expt_dir + "/current_val_metric.txt", 'w+') as f:
            f.write(f"{top1_avg.item()}\n")
    # ------------------------------------------------------------------------------------------------------------------


def train_one_epoch(
    train_loader_ft,
    model,
    criterion_ft,
    optimizer_ft,
    epoch,
    total_iter,
    config,
    writer,
    meters,
    ):
    # general meters
    batch_time_meter = meters["batch_time_meter"]
    data_time_meter = meters["data_time_meter"]
    losses_pt_meter = meters["losses_pt_meter"]
    losses_ft_meter = meters["losses_ft_meter"]
    top1_meter = meters["top1_meter"]
    top5_meter = meters["top5_meter"]
    
    
    norm_ft_meter_global = meters["norm_ft_meter_global"]
    target_std_meter = meters["target_std_meter"]
    norm_ft_avg_meter_lw = meters["norm_ft_avg_meter_lw"]
    norm_ft_std_meter_lw = meters["norm_ft_std_meter_lw"]
    
    meters_to_print = [
        batch_time_meter,
        losses_pt_meter,
        losses_ft_meter,
        top1_meter,
        target_std_meter,
        ]
    
    progress = ProgressMeter(
        num_batches=len(train_loader_ft),
        meters=meters_to_print,
        prefix=f"Epoch: [{epoch}]"
        )
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader_ft):
        
        total_iter += 1
        
        if config.expt.gpu is not None:
            images = images.cuda(config.expt.gpu, non_blocking=True)
        target = target.cuda(config.expt.gpu, non_blocking=True)

        get_gradients = False if config.expt.is_non_grad_based else True  # default is True

        loss_ft, backbone_grads_ft_lw, backbone_grads_ft_global = finetune(model, images, target, criterion_ft, optimizer_ft, losses_ft_meter, top1_meter, top5_meter, get_gradients=get_gradients)

        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_ft_lw, backbone_grads_ft=None, metric_type="norm")
        norm_ft_avg_meter_lw.update(mean), norm_ft_std_meter_lw.update(std)

        norm_ft_meter_global.update(torch.linalg.norm(backbone_grads_ft_global, 2))
        
        main_stats_meters = [
            norm_ft_meter_global,
            norm_ft_avg_meter_lw,
            target_std_meter,
            ]

        additional_stats_meters = [
            norm_ft_std_meter_lw
            ]

        meters_to_plot = {
            "main_meters":             main_stats_meters,
            "additional_stats_meters": additional_stats_meters
            }

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        end = time.time()

        if i % config.expt.print_freq == 0:
            progress.display(i)
        # write log epoch wise
        if config.expt.rank == 0 and i % config.expt.write_summary_frequency:
            write_to_summary_writer(
                total_iter=total_iter, loss_pt=None, loss_ft=loss_ft, data_time=data_time_meter, batch_time=batch_time_meter, optimizer_pt=None,
                optimizer_ft=optimizer_ft, top1=top1_meter, top5=top5_meter, meters_to_plot=meters_to_plot, writer=writer
                )
            
    return total_iter


def finetune(model, images_ft, target_ft, criterion_ft, optimizer_ft, losses_ft_meter, top1_meter, top5_meter, get_gradients=False):
    backbone_grads_lw = OrderedDict()
    backbone_grads_global = torch.Tensor().cuda()
    
    # fine-tuning
    model.eval()
    
    optimizer_ft.zero_grad()
    # in finetuning mode, we only optimize the classifier head's parameters
    # -> turn on backbone params grad computation before forward is called
    if get_gradients:
        model.module.backbone.requires_grad_(True)
    else:
        model.module.backbone.requires_grad_(False)
    
    model.module.classifier_head.requires_grad_(True)
    
    # compute outputs
    output_ft = model(images_ft, finetuning=True)
    loss_ft = criterion_ft(output_ft, target_ft)
    loss_ft.backward()
    
    if get_gradients:
        for key, param in model.module.backbone.named_parameters():
            grad_tensor = param.grad.detach_().clone().flatten()
            backbone_grads_lw[key] = torch.tensor(grad_tensor)
            backbone_grads_global = torch.cat([backbone_grads_global, grad_tensor], dim=0)
    
    # compute losses and measure accuracy
    acc1, acc5 = accuracy(output_ft, target_ft, topk=(1, 5))
    losses_ft_meter.update(loss_ft.item(), images_ft.size(0))
    top1_meter.update(acc1[0], images_ft.size(0))
    top5_meter.update(acc5[0], images_ft.size(0))
    
    # only optimizes classifier head parameters
    optimizer_ft.step()

    model.module.backbone.requires_grad_(False)
    # just to make sure to prevent grad leakage
    for param in model.module.parameters():
        param.grad = None
    
    return loss_ft, backbone_grads_lw, backbone_grads_global


if __name__ == '__main__':
    user = os.environ.get('USER')
    
    config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config file path.')
    config_parser.add_argument('--config', default="metassl/default_metassl_config.yaml", help='Select which yaml file to use depending on the selected experiment mode')
    parser = ArgumentParser()
    
    parser.add_argument('--expt', default="expt", type=str, metavar='N')
    parser.add_argument('--expt.expt_name', default='pre-training-fix-lr-100-256', type=str, help='experiment name')
    parser.add_argument('--expt.expt_mode', default='ImageNet', choices=["ImageNet", "CIFAR10", "ImageNet_BOHB", "CIFAR10_BOHB"], help='Define which dataset to use to select the correct yaml file.')
    parser.add_argument('--expt.save_model', action='store_false', help='save the model to disc or not (default: True)')
    parser.add_argument('--expt.save_model_frequency', default=10, type=int, metavar='N', help='save model frequency in # of epochs')
    parser.add_argument('--expt.ssl_model_checkpoint_path', type=str, help='path to the pre-trained model, resumes training if model with same config exists')
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
    parser.add_argument('--expt.is_non_grad_based', action='store_true', help='Set this flag to run default SimSiam or BOHB runs')
    parser.add_argument('--expt.warmup_epochs', default=10, type=int, metavar='N', help='denotes the number of epochs that we only pre-train without finetuning afterwards; warmup is turned off when set to 0; we use a linear incremental schedule during warmup')
    parser.add_argument('--expt.warmup_multiplier', default=2., type=float, metavar='N', help='A factor that is multiplied with the pretraining lr used in the linear incremental learning rate scheduler during warmup. The final lr is multiplier * pre-training lr')
    parser.add_argument('--expt.use_fix_aug_params', action='store_true', help='Use this flag if you want to try out specific aug params (e.g., from a best BOHB config). Default values will be overwritten then without crashing other experiments.')
    parser.add_argument('--expt.data_augmentation_mode', default='default', choices=['default', 'probability_augment', 'rand_augment'], help="Select which data augmentation to use. Default is for the standard SimSiam setting and for parameterize aug setting.")
    parser.add_argument('--expt.write_summary_frequency', default=10, type=int, metavar='N', help='Specifies, after how many batches the TensorBoard summary writer should flush new data to the summary object.')
    parser.add_argument('--expt.wd_decay_pt', action="store_true", help='use weight decay decay (annealing) during pre-training? (default: False)')
    parser.add_argument('--expt.wd_decay_ft', action="store_true", help='use weight decay decay (annealing) during fine-tuning? (default: False)')
    
    parser.add_argument('--train', default="train", type=str, metavar='N')
    parser.add_argument('--train.batch_size', default=256, type=int, metavar='N', help='in distributed setting this is the total batch size, i.e. batch size = individual bs * number of GPUs')
    parser.add_argument('--train.epochs', default=100, type=int, metavar='N', help='number of pre-training epochs')
    parser.add_argument('--train.start_epoch', default=0, type=int, metavar='N', help='start training at epoch n')
    parser.add_argument('--train.optimizer', type=str, default='sgd', help='optimizer type, options: sgd')
    parser.add_argument('--train.schedule', type=str, default='cosine', help='learning rate schedule, not implemented')
    parser.add_argument('--train.weight_decay', default=0.0001, type=float, metavar='N')
    parser.add_argument('--train.momentum', default=0.9, type=float, metavar='N', help='SGD momentum')
    parser.add_argument('--train.lr', default=0.05, type=float, metavar='N', help='pre-training learning rate')
    parser.add_argument('--train.wd_start', default=1e-3, type=float, help='Upper value of WD Decay. Only used when wd_decay is True.')
    parser.add_argument('--train.wd_end', default=1e-6, type=float, help='Lower value of WD Decay. Only used when wd_decay is True.')
    
    parser.add_argument('--finetuning', default="finetuning", type=str, metavar='N')
    parser.add_argument('--finetuning.batch_size', default=256, type=int, metavar='N', help='in distributed setting this is the total batch size, i.e. batch size = individual bs * number of GPUs')
    parser.add_argument('--finetuning.epochs', default=100, type=int, metavar='N', help='number of pre-training epochs')
    parser.add_argument('--finetuning.start_epoch', default=0, type=int, metavar='N', help='start training at epoch n')
    parser.add_argument('--finetuning.optimizer', type=str, default='sgd', help='optimizer type, options: sgd')
    parser.add_argument('--finetuning.schedule', type=str, default='cosine', help='learning rate schedule, not implemented')
    parser.add_argument('--finetuning.weight_decay', default=0.0, type=float, metavar='N')
    parser.add_argument('--finetuning.momentum', default=0.9, type=float, metavar='N', help='SGD momentum')
    parser.add_argument('--finetuning.lr', default=100, type=float, metavar='N', help='finetuning learning rate')
    parser.add_argument('--finetuning.valid_size', default=0.0, type=float, help='If valid_size > 0, pick some images from the trainset to do evaluation on. If valid_size=0 evaluation is done on the testset.')
    parser.add_argument('--finetuning.data_augmentation', default='none', choices=['none', 'p_probability_augment_pt', 'p_probability_augment_ft', 'p_probability_augment_1-pt'], help='Select if and how finetuning gets augmented.')
    parser.add_argument('--finetuning.wd_start', default=1e-3, type=float, help='Upper value of WD Decay. Only used when wd_decay is True.')
    parser.add_argument('--finetuning.wd_end', default=1e-6, type=float, help='Lower value of WD Decay. Only used when wd_decay is True.')
    
    parser.add_argument('--model', default="model", type=str, metavar='N')
    parser.add_argument('--model.model_type', type=str, default='resnet50', help='all torchvision ResNets')
    parser.add_argument('--model.seed', type=int, default=123, help='the seed')
    parser.add_argument('--model.turn_off_bn', action='store_true', help='turns off all batch norm instances in the model')
    
    parser.add_argument('--data', default="data", type=str, metavar='N')
    parser.add_argument('--data.seed', type=int, default=123, help='the seed')
    parser.add_argument('--data.dataset', type=str, default="ImageNet", help='supported datasets: CIFAR10, CIFAR100, ImageNet')
    parser.add_argument('--data.dataset_percentage_usage', type=float, default=100.0, help='Indicates what percentage of the data is used for the experiments.')
    
    parser.add_argument('--simsiam', default="simsiam", type=str, metavar='N')
    parser.add_argument('--simsiam.dim', type=int, default=2048, help='the feature dimension')
    parser.add_argument('--simsiam.pred_dim', type=int, default=512, help='the hidden dimension of the predictor')
    parser.add_argument('--simsiam.fix_pred_lr', action="store_false", help='fix learning rate for the predictor (default: True')
    
    parser.add_argument('--learnaug', default="learnaug", type=str, metavar='N')
    parser.add_argument('--learnaug.type', default="default", choices=["colorjitter", "default", "full_net"], help='Define which type of learned augmentation to use.')
    
    # parser.add_argument('--bohb', default="bohb", type=str, metavar='N')
    # parser.add_argument("--bohb.run_id", default="default_BOHB")
    # parser.add_argument("--bohb.seed", type=int, default=123, help="random seed")
    # parser.add_argument("--bohb.n_iterations", type=int, default=10, help="How many BOHB iterations")
    # parser.add_argument("--bohb.min_budget", type=int, default=2)
    # parser.add_argument("--bohb.max_budget", type=int, default=4)
    # parser.add_argument("--bohb.budget_mode", type=str, default="epochs", choices=["epochs", "data"], help="Choose your desired fidelity")
    # parser.add_argument("--bohb.eta", type=int, default=2)
    # parser.add_argument("--bohb.configspace_mode", type=str, default='color_jitter_strengths', choices=["imagenet_probability_simsiam_augment", "cifar10_probability_simsiam_augment", "color_jitter_strengths", "rand_augment", "probability_augment", "double_probability_augment"],
    # help='Define which configspace to use.')
    # parser.add_argument("--bohb.nic_name", default="lo", help="The network interface to use")  # local: "lo", cluster: "eth0"
    # parser.add_argument("--bohb.port", type=int, default=0)
    # parser.add_argument("--bohb.worker", action="store_true", help="Make this execution a worker server")
    # parser.add_argument("--bohb.warmstarting", type=bool, default=False)
    # parser.add_argument("--bohb.warmstarting_dir", type=str, default=None)
    # parser.add_argument("--bohb.test_env", action='store_true', help='If using this flag, the master runs a worker in the background and workers are not being shutdown after registering results.')

    parser.add_argument("--use_fixed_args", action="store_true", help="Flag to control whether to take arguments from yaml file as default or from arg parse")
    
    config = _parse_args(config_parser, parser)
    config = AttrDict(jsonargparse.namespace_to_dict(config))
    print("\n\n\n\nConfig:\n", config, "\n\n\n\n")
    
    # Check whether it is a BOHB run or not + organize expt_dir accordingly
    is_bohb_run = True if config.expt.expt_mode.endswith("BOHB") else False
    expt_dir = organize_experiment_saving(user=user, config=config, is_bohb_run=is_bohb_run)

    # Error check
    if config.finetuning.data_augmentation != "none" and config.expt.data_augmentation_mode != 'probability_augment':
        raise ValueError("If you use data augmentation for finetuning, 'probability_augment' is required as the data_augmentation_mode!")
    if is_bohb_run:
        assert config.finetuning.valid_size > 0.0, "BOHB requires a valid_size > 0.0"
        if config.bohb.configspace_mode == 'probability_augment' and config.expt.data_augmentation_mode != 'probability_augment':
            raise ValueError("If you run a BOHB experiment with 'probability_augment' configspace mode, you also need to select 'probability_augment' as data augmentation mode!")
        if config.bohb.configspace_mode == 'double_probability_augment' and config.finetuning.data_augmentation != "p_probability_augment_ft":
            raise ValueError("If you run a BOHB experiment with 'double_probability_augment' configspace mode, you also need to select 'p_probability_augment_ft' as finetuning data augmentation mode!")

    
    # Run BOHB / main
    if is_bohb_run:
        from metassl.hyperparameter_optimization.master import start_bohb_master
        
        start_bohb_master(yaml_config=config, expt_dir=expt_dir)
    
    else:
        main(config=config, expt_dir=expt_dir)
