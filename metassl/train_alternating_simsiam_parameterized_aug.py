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
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from jsonargparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.torch_utils import (hist_to_image,
                               tensor_to_image,
                               get_newest_model,
                               check_and_save_checkpoint,
                               deactivate_bn,
                               calc_all_layer_wise_stats,
                               validate,
                               accuracy,
                               get_sample_logprob)


warnings.filterwarnings("ignore", category=UserWarning)

try:
    # For execution in PyCharm
    from metassl.utils.data import get_train_valid_loader, get_test_loader, get_loaders, normalize_imagenet
    from metassl.utils.config import AttrDict, _parse_args
    from metassl.utils.meters import AverageMeter, ProgressMeter, ExponentialMovingAverageMeter
    from metassl.utils.simsiam_alternating import SimSiam
    import metassl.models.resnet_cifar as our_cifar_resnets
    from metassl.utils.simsiam import TwoCropsTransform, GaussianBlur
    from metassl.utils.augment import create_transforms, augment_per_image
except ImportError:
    # For execution in command line
    from .utils.data import get_train_valid_loader, get_test_loader, get_loaders, normalize_imagenet
    from .utils.config import AttrDict, _parse_args
    from .utils.meters import AverageMeter, ProgressMeter, ExponentialMovingAverageMeter
    from .utils.simsiam_alternating import SimSiam
    from .utils.simsiam import TwoCropsTransform, GaussianBlur
    from .models import resnet_cifar as our_cifar_resnets
    from .utils.augment import create_transforms, augment_per_image

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
    )

# augmentation strengths
color_jitter_strengths_brightness = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
color_jitter_strengths_contrast = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
color_jitter_strengths_saturation = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
color_jitter_strengths_hue = [0.0, 0.1, 0.2, 0.3, 0.4]

# histograms
color_jitter_histogram_brightness = {k: 0 for k in color_jitter_strengths_brightness}
color_jitter_histogram_contrast = {k: 0 for k in color_jitter_strengths_contrast}
color_jitter_histogram_saturation = {k: 0 for k in color_jitter_strengths_saturation}
color_jitter_histogram_hue = {k: 0 for k in color_jitter_strengths_hue}

color_jitter_hists = {
    "b": color_jitter_histogram_brightness,
    "c": color_jitter_histogram_contrast,
    "s": color_jitter_histogram_saturation,
    "h": color_jitter_histogram_hue
    }

color_jitter_strengths = {
    "b": color_jitter_strengths_brightness,
    "c": color_jitter_strengths_contrast,
    "s": color_jitter_strengths_saturation,
    "h": color_jitter_strengths_hue
    }


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
    
    # todo: check backpack + ddp + resnet with sam; backpack raises errors when using inplace operations
    if config.expt.image_wise_gradients:
        for module in model.modules():
            if hasattr(module, "inplace"):
                module.inplace = False
    
    if config.model.turn_off_bn:
        print("Turning off BatchNorm in entire model.")
        deactivate_bn(model)
        model.encoder_head[6].bias.requires_grad = True
    
    # infer learning rate before changing batch size
    init_lr_pt = config.train.lr * config.train.batch_size / 256
    init_lr_ft = config.finetuning.lr * config.finetuning.batch_size / 256
    
    config.train.init_lr_pt = init_lr_pt
    config.train.init_lr_ft = init_lr_ft
    
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
            
            if config.expt.image_wise_gradients:
                model = extend(model)
                print("using backpack")
            
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
    
    aug_w_b = torch.zeros(len(color_jitter_strengths_brightness), requires_grad=True)
    aug_w_c = torch.zeros(len(color_jitter_strengths_contrast), requires_grad=True)
    aug_w_s = torch.zeros(len(color_jitter_strengths_saturation), requires_grad=True)
    aug_w_h = torch.zeros(len(color_jitter_strengths_hue), requires_grad=True)
    
    bound = 1. / math.sqrt(aug_w_b.size(0))
    bound_h = 1. / math.sqrt(aug_w_h.size(0))
    
    nn.init.uniform(aug_w_b, -bound, bound)
    nn.init.uniform(aug_w_c, -bound, bound)
    nn.init.uniform(aug_w_s, -bound, bound)
    nn.init.uniform(aug_w_h, -bound_h, bound_h)
    
    # color_jitter_dist = torch.distributions.Categorical(probs=torch.softmax(aug_w, dim=0))
    optimizer_aug = torch.optim.Adam([aug_w_b, aug_w_c, aug_w_s, aug_w_h], 0.001)
    
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
            optimizer_aug.load_state_dict(checkpoint['optimizer_aug'])
            total_iter = checkpoint['total_iter']
            print(f"=> loaded checkpoint '{config.expt.ssl_model_checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{config.expt.ssl_model_checkpoint_path}'")
    
    # Data loading code
    traindir = os.path.join(config.data.dataset, 'train')
    
    train_loader_pt, train_sampler_pt, train_loader_ft, train_sampler_ft, test_loader_ft = get_loaders(traindir, config, parameterize_augmentation=True)
    
    cudnn.benchmark = True
    writer = None
    
    if config.expt.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(expt_dir, "tensorboard"))
    
    if config.expt.layer_wise_stats:
        meter_name = "Cos. Sim. PT-FT layer-w. average"
    else:
        meter_name = "Cos. Sim. PT-FT"
    
    cos_sim_ema_meter = ExponentialMovingAverageMeter(meter_name, window=100, alpha=2, fmt=':6.4f')
    
    for epoch in range(config.train.start_epoch, config.train.epochs):
        
        if config.expt.distributed:
            train_sampler_pt.set_epoch(epoch)
            train_sampler_ft.set_epoch(epoch)
        
        warmup = config.train.warmup > epoch
        print(f"Warmup status: {warmup}")
        
        if warmup:
            cur_lr_pt = adjust_learning_rate(optimizer_pt, init_lr_pt, epoch, total_epochs=config.train.warmup, warmup=True)
            print(f"current warmup lr: {cur_lr_pt}")
        else:
            cur_lr_pt = adjust_learning_rate(optimizer_pt, init_lr_pt, epoch, config.train.epochs)
            
        cur_lr_ft = adjust_learning_rate(optimizer_ft, init_lr_ft, epoch, config.finetuning.epochs)
        print(f"current pretrain lr: {cur_lr_pt}, finetune lr: {cur_lr_ft}")
        
        total_iter = train_one_epoch(
            train_loader_pt=train_loader_pt,
            train_loader_ft=train_loader_ft,
            model=model,
            criterion_pt=criterion_pt,
            criterion_ft=criterion_ft,
            optimizer_pt=optimizer_pt,
            optimizer_ft=optimizer_ft,
            optimizer_aug=optimizer_aug,
            aug_w_b=aug_w_b,
            aug_w_c=aug_w_c,
            aug_w_s=aug_w_s,
            aug_w_h=aug_w_h,
            epoch=epoch,
            total_iter=total_iter,
            config=config,
            writer=writer,
            warmup=warmup,
            layer_wise_stats=config.expt.layer_wise_stats,
            cos_sim_ema_meter=cos_sim_ema_meter,
            )
        
        # evaluate on validation set
        if epoch % config.expt.eval_freq == 0:
            top1_avg = validate(test_loader_ft, model, criterion_ft, config, finetuning=True)
            if config.expt.rank == 0:
                writer.add_scalar('Test/Accuracy@1', top1_avg, total_iter)
        
        check_and_save_checkpoint(config, ngpus_per_node, total_iter, epoch, model, optimizer_pt, optimizer_ft, expt_dir, optimizer_aug=optimizer_aug)
    
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
    optimizer_aug,
    aug_w_b,
    aug_w_c,
    aug_w_s,
    aug_w_h,
    epoch,
    total_iter,
    config,
    writer,
    warmup=False,
    layer_wise_stats=False,
    cos_sim_ema_meter=None,
    ):
    batch_time_meter = AverageMeter('Time', ':6.3f')
    data_time_meter = AverageMeter('Data', ':6.3f')
    losses_pt_meter = AverageMeter('Loss PT', ':.4f')
    losses_ft_meter = AverageMeter('Loss FT', ':.4e')
    top1_meter = AverageMeter('Acc@1', ':6.2f')
    top5_meter = AverageMeter('Acc@5', ':6.2f')

    norm_aug_brightness_grad_meter = AverageMeter('Norm Aug. brightness gradient', ':6.4f')
    norm_aug_contrast_grad_meter = AverageMeter('Norm Aug. contrast gradient', ':6.4f')
    norm_aug_saturation_grad_meter = AverageMeter('Norm Aug. saturation gradient', ':6.4f')
    norm_aug_hue_grad_meter = AverageMeter('Norm Aug. hue gradient', ':6.4f')
    
    if layer_wise_stats:
        if not cos_sim_ema_meter:
            cos_sim_ema_meter = ExponentialMovingAverageMeter('Cos. Sim. PT-FT layer-w.', window=100, alpha=2, fmt=':6.4f')
        cos_sim_std_meter = AverageMeter('Cos. Sim. PT-FT layer-w. std.', ':6.4f')
        
        dot_prod_avg_meter = AverageMeter('Dot Product PT-FT layer-w. average', ':6.4f')
        dot_prod_std_meter = AverageMeter('Dot Product PT-FT layer-w. std.', ':6.4f')
        eucl_dis_avg_meter = AverageMeter('Eucl. Dist. PT-FT layer-w. average', ':6.4f')
        eucl_dis_std_meter = AverageMeter('Eucl. Dist. PT-FT layer-w. std.', ':6.4f')
        norm_pt_avg_meter = AverageMeter('Norm PT layer-w. average', ':6.4f')
        norm_pt_std_meter = AverageMeter('Norm PT layer-w. std.', ':6.4f')
        norm_ft_avg_meter = AverageMeter('Norm FT layer-w. average', ':6.4f')
        norm_ft_std_meter = AverageMeter('Norm FT layer-w. std.', ':6.4f')
        
        meters = [batch_time_meter, losses_pt_meter, losses_ft_meter, top1_meter, cos_sim_ema_meter, dot_prod_avg_meter, eucl_dis_avg_meter, norm_pt_avg_meter, norm_ft_avg_meter]
    else:
        if not cos_sim_ema_meter:
            cos_sim_ema_meter = ExponentialMovingAverageMeter('Cos. Sim. PT-FT', window=100, alpha=2, fmt=':6.4f')
        
        dot_prod_meter = AverageMeter('Dot Product PT-FT', ':6.4f')
        eucl_dis_meter = AverageMeter('Eucl. Dist. PT-FT', ':6.4f')
        norm_pt_meter = AverageMeter('Norm PT', ':6.4f')
        norm_ft_meter = AverageMeter('Norm FT', ':6.4f')
        
        meters = [batch_time_meter, losses_pt_meter, losses_ft_meter, top1_meter, cos_sim_ema_meter, dot_prod_meter, eucl_dis_meter, norm_pt_meter, norm_ft_meter]
    
    progress = ProgressMeter(
        num_batches=len(train_loader_pt),
        meters=meters,
        prefix=f"Epoch: [{epoch}]"
        )
    
    end = time.time()
    assert len(train_loader_pt) <= len(train_loader_ft), 'So since this seems to break, we should write code to run multiple finetune epoch per pretrain epoch'
    for i, ((images_pt, _), (images_ft, target_ft)) in enumerate(zip(train_loader_pt, train_loader_ft)):
        
        total_iter += 1
        
        color_jitter_action_idx_b, color_jitter_logprob_b, _ = get_sample_logprob(logits=aug_w_b)
        color_jitter_action_idx_c, color_jitter_logprob_c, _ = get_sample_logprob(logits=aug_w_c)
        color_jitter_action_idx_s, color_jitter_logprob_s, _ = get_sample_logprob(logits=aug_w_s)
        color_jitter_action_idx_h, color_jitter_logprob_h, _ = get_sample_logprob(logits=aug_w_h)
        
        strength_b = color_jitter_strengths_brightness[color_jitter_action_idx_b]
        strength_c = color_jitter_strengths_contrast[color_jitter_action_idx_c]
        strength_s = color_jitter_strengths_saturation[color_jitter_action_idx_s]
        strength_h = color_jitter_strengths_hue[color_jitter_action_idx_h]
        
        color_jitter_histogram_brightness[strength_b] += 1
        color_jitter_histogram_contrast[strength_c] += 1
        color_jitter_histogram_saturation[strength_s] += 1
        color_jitter_histogram_hue[strength_h] += 1
        
        if config.expt.rank == 0 and i % (config.expt.print_freq * 100) == 0:
            rand_int = torch.randint(high=images_pt.shape[0], size=(1,))
            untransformed_image = torch.permute(images_pt[rand_int].squeeze(), (1, 2, 0)).cpu()
            
        if config.expt.gpu is not None:
            images_pt = images_pt.cuda(config.expt.gpu, non_blocking=True)
        
        if config.expt.image_wise_gradients:
            parameterized_transform_list = create_transforms(strength_b, strength_c, strength_s, strength_h, image_height=images_pt.shape[2], image_width=images_pt.shape[3])
            images_pt = augment_per_image(parameterized_transform_list, images_pt)
        else:
            parameterized_transform = create_transforms(strength_b, strength_c, strength_s, strength_h, image_height=images_pt.shape[2], image_width=images_pt.shape[3])[0]
            images_pt = parameterized_transform(images_pt)
        
        if config.expt.gpu is not None:
            images_pt[0] = images_pt[0].contiguous()
            images_pt[1] = images_pt[1].contiguous()
            images_ft = images_ft.cuda(config.expt.gpu, non_blocking=True)
            target_ft = target_ft.cuda(config.expt.gpu, non_blocking=True)

        if config.expt.rank == 0 and i % (config.expt.print_freq * 100) == 0:
            img0 = torch.permute(images_pt[0][rand_int].squeeze(), (1, 2, 0)).cpu()
            img1 = torch.permute(images_pt[1][rand_int].squeeze(), (1, 2, 0)).cpu()
            images_pt_to_plot = [untransformed_image, img0, img1]
        
        loss_pt, backbone_grads_pt = pretrain(model, images_pt, criterion_pt, optimizer_pt, losses_pt_meter, data_time_meter, end, config=config, alternating_mode=True, layer_wise_stats=layer_wise_stats)
        
        backbone_grads_ft = None
        if not warmup:
            loss_ft, backbone_grads_ft = finetune(model, images_ft, target_ft, criterion_ft, optimizer_ft, losses_ft_meter, top1_meter, top5_meter, config=config, alternating_mode=True, layer_wise_stats=layer_wise_stats)
        else:
            losses_ft_meter.update(np.inf)
            loss_ft = np.inf
        
        if not warmup:
            if layer_wise_stats:
                calc_all_layer_wise_stats(
                    backbone_grads_pt=backbone_grads_pt,
                    backbone_grads_ft=backbone_grads_ft,
                    cos_sim_ema_meter=cos_sim_ema_meter,
                    cos_sim_std_meter=cos_sim_std_meter,
                    dot_prod_avg_meter=dot_prod_avg_meter,
                    dot_prod_std_meter=dot_prod_std_meter,
                    eucl_dis_avg_meter=eucl_dis_avg_meter,
                    eucl_dis_std_meter=eucl_dis_std_meter,
                    norm_pt_avg_meter=norm_pt_avg_meter,
                    norm_pt_std_meter=norm_pt_std_meter,
                    norm_ft_avg_meter=norm_ft_avg_meter,
                    norm_ft_std_meter=norm_ft_std_meter,
                    warmup=False,
                    )
            else:
                cos_sim_ema_meter.update(F.cosine_similarity(backbone_grads_pt, backbone_grads_ft, dim=0))
                dot_prod_meter.update(torch.dot(backbone_grads_pt, backbone_grads_ft))
                eucl_dis_meter.update(torch.linalg.norm(backbone_grads_pt - backbone_grads_ft, 2))
                norm_pt_meter.update(torch.linalg.norm(backbone_grads_pt, 2))
                norm_ft_meter.update(torch.linalg.norm(backbone_grads_ft, 2))
            
            optimizer_aug.zero_grad()
            reward = (cos_sim_ema_meter.val - cos_sim_ema_meter.ema)
            
            color_jitter_logprob_b = -(color_jitter_logprob_b * reward)
            color_jitter_logprob_b.backward()
            
            color_jitter_logprob_c = -(color_jitter_logprob_c * reward)
            color_jitter_logprob_c.backward()
            
            color_jitter_logprob_s = -(color_jitter_logprob_s * reward)
            color_jitter_logprob_s.backward()
            
            color_jitter_logprob_h = -(color_jitter_logprob_h * reward)
            color_jitter_logprob_h.backward()
            
            optimizer_aug.step()
            norm_aug_brightness_grad_meter.update(torch.linalg.norm(aug_w_b.grad.data, 2))
            norm_aug_contrast_grad_meter.update(torch.linalg.norm(aug_w_c.grad.data, 2))
            norm_aug_saturation_grad_meter.update(torch.linalg.norm(aug_w_s.grad.data, 2))
            norm_aug_hue_grad_meter.update(torch.linalg.norm(aug_w_h.grad.data, 2))
        
        else:
            if layer_wise_stats:
                calc_all_layer_wise_stats(
                    backbone_grads_pt=backbone_grads_pt,
                    backbone_grads_ft=backbone_grads_ft,
                    cos_sim_ema_meter=cos_sim_ema_meter,
                    cos_sim_std_meter=cos_sim_std_meter,
                    dot_prod_avg_meter=dot_prod_avg_meter,
                    dot_prod_std_meter=dot_prod_std_meter,
                    eucl_dis_avg_meter=eucl_dis_avg_meter,
                    eucl_dis_std_meter=eucl_dis_std_meter,
                    norm_pt_avg_meter=norm_pt_avg_meter,
                    norm_pt_std_meter=norm_pt_std_meter,
                    norm_ft_avg_meter=norm_ft_avg_meter,
                    norm_ft_std_meter=norm_ft_std_meter,
                    warmup=True
                    )
            else:
                # no resetting needed, as meters are freshly initialized at each epoch
                cos_sim_ema_meter.update(0.)
                dot_prod_meter.update(0.)
                eucl_dis_meter.update(0.)
                norm_pt_meter.update(torch.linalg.norm(backbone_grads_pt, 2))
                norm_ft_meter.update(0.)
                norm_aug_brightness_grad_meter.update(0.)
        
        if layer_wise_stats:
            advanced_stats_meters = [cos_sim_ema_meter, dot_prod_avg_meter, dot_prod_std_meter, eucl_dis_avg_meter, eucl_dis_std_meter, norm_pt_avg_meter, norm_pt_std_meter, norm_ft_avg_meter, norm_ft_std_meter, norm_aug_brightness_grad_meter]
        else:
            advanced_stats_meters = [cos_sim_ema_meter, dot_prod_meter, eucl_dis_meter, norm_pt_meter, norm_ft_meter]
        
        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        end = time.time()
        
        if i % config.expt.print_freq == 0:
            progress.display(i)
        if config.expt.rank == 0:
            write_to_summary_writer(total_iter, loss_pt, loss_ft, data_time_meter, batch_time_meter, optimizer_pt, optimizer_ft, top1_meter, top5_meter, advanced_stats_meters, writer)
        # expensive stats
        if config.expt.rank == 0 and i % (config.expt.print_freq * 100) == 0:
            img = hist_to_image(color_jitter_histogram_brightness, "Color Jitter Strength Brightness Counts")
            writer.add_image(tag="Advanced Stats/color jitter strength brightness", img_tensor=img, global_step=total_iter)
            
            img = hist_to_image(color_jitter_histogram_contrast, "Color Jitter Strength Contrast Counts")
            writer.add_image(tag="Advanced Stats/color jitter strength contrast", img_tensor=img, global_step=total_iter)
            
            img = hist_to_image(color_jitter_histogram_saturation, "Color Jitter Strength Saturation Counts")
            writer.add_image(tag="Advanced Stats/color jitter strength saturation", img_tensor=img, global_step=total_iter)
            
            img = hist_to_image(color_jitter_histogram_hue, "Color Jitter Strength Hue Counts")
            writer.add_image(tag="Advanced Stats/color jitter strength hue", img_tensor=img, global_step=total_iter)

            img = tensor_to_image(images_pt_to_plot[0], "Randomly sampled untransformed image")
            writer.add_image(tag="Advanced Stats/sampled untransformed image 1", img_tensor=img, global_step=total_iter)
            
            img = tensor_to_image(images_pt_to_plot[1], "Randomly sampled transformed image 1")
            writer.add_image(tag="Advanced Stats/sampled transformed image 1", img_tensor=img, global_step=total_iter)

            img = tensor_to_image(images_pt_to_plot[2], "Randomly sampled transformed image 2")
            writer.add_image(tag="Advanced Stats/sampled transformed image 2", img_tensor=img, global_step=total_iter)
    
    return total_iter


def pretrain(model, images_pt, criterion_pt, optimizer_pt, losses_pt, data_time, end, config, alternating_mode=False, layer_wise_stats=False):
    if layer_wise_stats:
        backbone_grads = OrderedDict()
    else:
        backbone_grads = torch.Tensor().cuda()
    
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
    if config.expt.image_wise_gradients:
        with backpack(BatchGrad()):
            loss_pt.backward()
    else:
        loss_pt.backward()
    # step does not change .grad field of the parameters.
    optimizer_pt.step()
    
    if alternating_mode:
        for key, param in model.module.backbone.named_parameters():
            if layer_wise_stats:
                backbone_grads[key] = torch.tensor(param.grad.detach_().clone().flatten())
            else:
                backbone_grads = torch.cat([backbone_grads, param.grad.detach_().clone().flatten()], dim=0)
    
    return loss_pt, backbone_grads


def finetune(model, images_ft, target_ft, criterion_ft, optimizer_ft, losses_ft_meter, top1_meter, top5_meter, config, alternating_mode=False, layer_wise_stats=False):
    if layer_wise_stats:
        backbone_grads = OrderedDict()
    else:
        backbone_grads = torch.Tensor().cuda()
    
    # fine-tuning
    model.eval()
    
    optimizer_ft.zero_grad()
    # in finetuning mode, we only optimize the classifier head's parameters
    # -> turn on backbone params grad computation before forward is called
    if alternating_mode:
        model.module.backbone.requires_grad_(True)
    else:
        model.module.backbone.requires_grad_(False)
    
    model.module.classifier_head.requires_grad_(True)
    
    # compute outputs
    output_ft = model(images_ft, finetuning=True)
    loss_ft = criterion_ft(output_ft, target_ft)
    if config.expt.image_wise_gradients:
        with backpack(BatchGrad()):
            loss_ft.backward()
    else:
        loss_ft.backward()
    
    if alternating_mode:
        for key, param in model.module.backbone.named_parameters():
            if layer_wise_stats:
                backbone_grads[key] = torch.tensor(param.grad.detach_().clone().flatten())
            else:
                backbone_grads = torch.cat([backbone_grads, param.grad.detach_().clone().flatten()], dim=0)
    
    # compute losses and measure accuracy
    acc1, acc5 = accuracy(output_ft, target_ft, topk=(1, 5))
    losses_ft_meter.update(loss_ft.item(), images_ft.size(0))
    top1_meter.update(acc1[0], images_ft.size(0))
    top5_meter.update(acc5[0], images_ft.size(0))
    
    # only optimizes classifier head parameters
    optimizer_ft.step()
    
    # just to make sure to prevent grad leakage
    for param in model.module.parameters():
        param.grad = None
    
    return loss_ft, backbone_grads


def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs, warmup=False):
    """Decay the learning rate based on schedule; during warmup, increment the learning rate linearly (not used for fixed lr)"""
    if warmup:
        cur_lr = init_lr * min(1., (float((epoch + 1) / total_epochs)))
    else:
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
        
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
            return init_lr
        else:
            param_group['lr'] = cur_lr
            return cur_lr


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
    parser.add_argument('--expt.layer_wise_stats', action='store_true', help='compute the advanced stats for each layer separately and then plot the average and deviation (default: False).')
    parser.add_argument('--expt.image_wise_gradients', action='store_true', help='compute image wise gradients with backpack (default: False).')
    
    parser.add_argument('--train', default="train", type=str, metavar='N')
    parser.add_argument('--train.batch_size', default=256, type=int, metavar='N', help='in distributed setting this is the total batch size, i.e. batch size = individual bs * number of GPUs')
    parser.add_argument('--train.epochs', default=100, type=int, metavar='N', help='number of pre-training epochs')
    parser.add_argument('--train.start_epoch', default=0, type=int, metavar='N', help='start training at epoch n')
    parser.add_argument('--train.optimizer', type=str, default='sgd', help='optimizer type, options: sgd')
    parser.add_argument('--train.schedule', type=str, default='cosine', help='learning rate schedule, not implemented')
    parser.add_argument('--train.warmup', default=0, type=int, metavar='N', help='denotes the number of epochs that we only pre-train without finetuning afterwards; warmup is turned off when set to 0; we use a linear incremental schedule during warmup')
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
