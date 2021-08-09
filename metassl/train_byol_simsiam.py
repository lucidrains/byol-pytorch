import warnings

warnings.filterwarnings('ignore')
import os
import pathlib
import random
import yaml

from byol_pytorch import BYOL

from argparse import ArgumentParser

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import torch
from torch import nn

# original torchvision resnet implementation:
from torchvision.models import resnet18, resnet50  # torchvision (for ImageNet)

from metassl.models.resnet import ResNet
from metassl.utils.data import get_test_loader, get_train_valid_loader
from metassl.utils.my_optimizer import MyOptimizer
from metassl.utils.summary import SummaryDict
from metassl.utils.supporter import Supporter
from metassl.utils.torch_utils import count_parameters

# custom implementation; faster but smaller complexity through fewer conv layers and smaller fc layer (for CIFAR10/100):
# from metassl.models.ResNet_CIFAR_small import resnet20, resnet56

# custom implementation; close to the original implementation with 3x3 instead of 7x7 conv1 and stride 1 (for CIFAR10/100):
# from metassl.models.ResNet_CIFAR import resnet18, resnet50

print("CUDA", torch.cuda.is_available())


def train_model(config, logger, checkpoint, local_rank):
    logger.print_config(config)
    
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    
    distributed = config.train.gpus > 1
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if distributed:
        device = torch.device(local_rank)
        torch.cuda.set_device(device)
    
    train_loader, valid_loader = get_train_valid_loader(
        data_dir=config.data.data_dir,
        batch_size=config.train.batch_size,
        random_seed=config.data.seed,
        dataset_name=config.data.dataset,
        num_workers=50,
        pin_memory=False,
        download=True,
        distributed=distributed,
        )
    
    test_loader = get_test_loader(
        data_dir=config.data.data_dir,
        batch_size=config.train.batch_size,
        shuffle=False,
        dataset_name=config.data.dataset,
        num_workers=50,
        pin_memory=False,
        download=True,
        )
    
    out_size = len(train_loader.dataset.classes)
    
    logger("device", device)
    logger("out_size", out_size)
    
    if config.model.model_type == "resnet18":
        model = resnet18(pretrained=False).to(device)
        # model = resnet20(num_classes=out_size).to(device)
    elif config.model.model_type == "resnet50":
        model = resnet50(num_classes=out_size).to(device)
        # model = resnet50(num_classes=out_size)
        # model = ResNet(config.data.dataset, 50, out_size, bottleneck=True).to(device)
        # model = ResNet(config.data.dataset, 50, out_size, bottleneck=True)
        # model = resnet56(num_classes=out_size).to(device)
    else:
        raise ValueError(f'Model not valid: {config.model.model_type}')

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    ### BYOL part ###
    model = BYOL(
        model,
        image_size=256,
        hidden_layer='avgpool',
        use_momentum=False  # turn off momentum in the target encoder
        )
    
    
    model = DDP(
        model.to(device),
        device_ids=[local_rank],
        find_unused_parameters=True,
        )
    
    logger.log("model_parameters", count_parameters(model.parameters()))
    
    summary_dict = SummaryDict()
    
    logger("START initial validation")
    
    valid_loss, accuracy = test(config, model, device, train_loader, valid_loader, local_rank)
    summary_dict["step"] = 0
    summary_dict["valid_loss"] = valid_loss
    summary_dict["valid_accuracy"] = accuracy
    summary_dict["train_loss"] = 0
    summary_dict["learning_rate"] = 0
    
    iter_per_epoch = len(train_loader)
    max_steps = config.train.epochs * iter_per_epoch
    logger("iter_per_epoch", iter_per_epoch)
    logger("max_steps", max_steps)
    
    optimizer = MyOptimizer(0, model.parameters(), max_steps, iter_per_epoch, **config.optim.get_dict)
    
    epoch_resume = 0
    if config.expt.resume_training:
        model, optimizer, epoch_resume, _ = checkpoint.load_newest_training(model, optimizer, logger)
    
    for epoch in range(epoch_resume, config.train.epochs):
        train_loader.sampler.set_epoch(epoch)
        
        logger(f"START epoch {epoch}")
        logger.start_timer("train")
        
        model.train()
        
        train_loss = []
        for step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            loss = model(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
            
            if step % config.train.eval_freq == 0:
                print(f"Epoch {epoch:{5}} / {config.train.epochs:{5}}, Loss {loss.item():.5f}")
        
        if config.expt.save_model and epoch % config.expt.save_model_freq == 0:
            checkpoint.save_training(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.get_state_dict(),
                epoch=epoch,
                loss=train_loss,
                number=epoch,
                )
        
        logger.timer("train", epoch)
        
        valid_loss, accuracy = test(model, device, valid_loader)
        summary_dict["step"] = epoch + 1
        summary_dict["train_loss"] = np.mean(train_loss)
        summary_dict["valid_loss"] = valid_loss
        summary_dict["valid_accuracy"] = accuracy
        summary_dict["learning_rate"] = optimizer._rate
        logger("train_loss", np.mean(train_loss), epoch)
        logger("valid_loss", valid_loss, epoch)
        logger("valid_accuracy", accuracy, epoch)
        logger("learning_rate", optimizer._rate, epoch)
        
        summary_dict.save(checkpoint.dir / "summary_dict.npy")
    
    # TEST FINAL MODEL
    # test_loss, test_accuracy = test(model, device, test_loader)
    # summary_dict["test_loss"] = test_loss
    # summary_dict["test_accuracy"] = test_accuracy
    # logger("test_loss", test_loss, epoch)
    # logger("test_accuracy", test_accuracy, epoch)
    
    summary_dict.save(checkpoint.dir / "summary_dict.npy")


def test(config, byol_model: BYOL, device, train_loader, test_loader, local_rank):
    byol_model.eval()
    model = byol_model.module.net
    model.fc = DDP(model.fc, device_ids=[local_rank], find_unused_parameters=True,)
    require_grads = []
    for p in model.parameters():
        require_grads.append(p.requires_grad)
        p.requires_grad_(False)
    
    for p in model.fc.parameters():
        p.requires_grad_(True)

    iter_per_epoch = len(train_loader)
    max_steps = config.finetuning.epochs * iter_per_epoch
    finetuning_optimizer = MyOptimizer(0, model.fc.parameters(), max_steps, iter_per_epoch, **config.finetuning.optim.dict)

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, config.finetuning.epochs):
        train_loader.sampler.set_epoch(epoch)
    
        #logger(f"START epoch {epoch}")
        #logger.start_timer("train")
    
    
        train_loss = []
        for step, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            y = model(data)
            l = criterion(y,target)
            finetuning_optimizer.zero_grad()
            l.backward()
            finetuning_optimizer.step()
            train_loss.append(l.item())
            print('finetune train loss',l)
            
    for p, rg in zip(model.parameters(), require_grads):
        p.requires_grad_(rg)
    
    
    
    
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= total
    accuracy = 100. * correct / total
    
    return test_loss, accuracy


def begin_training(config, expt_dir):
    expt_dir = pathlib.Path(expt_dir)
    
    if config['expt']['resume_training']:
        reload_expt = True
    else:
        reload_expt = False
    
    supporter = Supporter(experiments_dir=expt_dir, config_dict=config, count_expt=True, reload_expt=reload_expt)
    config = supporter.get_config()
    logger = supporter.get_logger()
    
    ### --- DDP --- ###
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")
        }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )
    
    train_model(config=config, logger=logger, checkpoint=supporter.ckp, local_rank=int(env_dict["LOCAL_RANK"]))
    
    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    user = os.environ.get('USER')
    
    with open("metassl/default_metassl_config.yaml", "r") as f:
        config = yaml.load(f)
    
    expt_dir = f"/home/{user}/workspace/experiments"
    config['data']['data_dir'] = f'/home/{user}/workspace/data/metassl'
    
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    begin_training(config=config, expt_dir=expt_dir)
