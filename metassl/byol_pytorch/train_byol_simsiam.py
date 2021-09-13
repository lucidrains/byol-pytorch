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
import torch.nn.functional as F

# original torchvision resnet implementation:
from torchvision.models import resnet18, resnet50  # torchvision (for ImageNet)

from metassl.utils.data import get_test_loader, get_train_valid_loader
from metassl.utils.my_optimizer import MyOptimizer
from metassl.utils.summary import SummaryDict
from metassl.utils.supporter import Supporter
from metassl.utils.torch_utils import count_parameters
from metassl.utils.torch_utils import knn_predict

# custom implementation; faster but smaller complexity through fewer conv layers and smaller fc layer (for CIFAR10/100):
# from metassl.models.ResNet_CIFAR_small import resnet20, resnet56

# custom implementation; close to the original implementation with 3x3 instead of 7x7 conv1 and stride 1 (for CIFAR10/100):
# from metassl.models.ResNet_CIFAR import resnet18, resnet50

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
        shuffle=True,
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

    if not distributed:
        logger("device", device)
        logger("out_size", out_size)
    elif distributed and local_rank == 0:
        logger("distributed mode; world size", torch.distributed.get_world_size(group=None))
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
    
    if not distributed or distributed and local_rank == 0:
        logger.log("model_parameters", count_parameters(model.parameters()))
        logger("START initial validation")
    
    summary_dict = SummaryDict()
    
    iter_per_epoch = len(train_loader)
    max_steps = config.train.epochs * iter_per_epoch
    
    # valid_loss, accuracy = test_linear_classification(config, model, device, train_loader, valid_loader, local_rank)
    if not distributed or distributed and local_rank == 0:
        accuracy = test_knn(model.module, device, train_loader, valid_loader, out_size, logger)
        summary_dict["step"] = 0
        summary_dict["valid_accuracy"] = accuracy
        summary_dict["train_loss"] = 0
        summary_dict["learning_rate"] = 0
    
        logger("iter_per_epoch", iter_per_epoch)
        logger("max_steps", max_steps)
    
    optimizer = MyOptimizer(0, model.parameters(), max_steps, iter_per_epoch, **config.optim.get_dict)
    
    epoch_resume = 0
    if config.expt.resume_training:
        model, optimizer, epoch_resume, _ = checkpoint.load_newest_training(model, optimizer, logger)
    
    for epoch in range(epoch_resume, config.train.epochs):
        train_loader.sampler.set_epoch(epoch)

        if not distributed or distributed and local_rank == 0:
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
            
            if not distributed and step % config.train.eval_freq == 0:
                print(f"Epoch {epoch:{5}} / {config.train.epochs:{5}}, Loss {loss.item():.5f}")
            elif distributed and step % config.train.eval_freq == 0:
                loss = loss.clone().detach()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()
                if dist.get_rank() == 0:
                    print(f"Epoch {epoch:{5}} / {config.train.epochs:{5}}, Loss {loss.item():.5f}")
        
        if not distributed and config.expt.save_model and epoch % config.expt.save_model_freq == 0 or \
            (distributed and config.expt.save_model and epoch % config.expt.save_model_freq == 0 and local_rank == 0):
            checkpoint.save_training(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.get_state_dict(),
                epoch=epoch,
                loss=train_loss,
                number=epoch,
                )

        if not distributed or distributed and local_rank == 0:
            logger.timer("train", epoch)
            # valid_loss, accuracy = test_linear_classification(model, device, valid_loader)
            accuracy = test_knn(model.module, device, train_loader, valid_loader, out_size, logger)
            summary_dict["step"] = epoch + 1
            summary_dict["train_loss"] = np.mean(train_loss)
            # summary_dict["valid_loss"] = valid_loss
            summary_dict["valid_accuracy"] = accuracy
            summary_dict["learning_rate"] = optimizer._rate
            logger("train_loss", np.mean(train_loss), epoch)
            # logger("valid_loss", valid_loss, epoch)
            logger("valid_accuracy", accuracy, epoch)
            logger("learning_rate", optimizer._rate, epoch)
            
            summary_dict.save(checkpoint.dir / "summary_dict.npy")
    
    # TEST FINAL MODEL
    # if not distributed or distributed and local_rank == 0:
    #     test_loss, test_accuracy = test(model, device, test_loader)
    #     summary_dict["test_loss"] = test_loss
    #     summary_dict["test_accuracy"] = test_accuracy
    #     logger("test_loss", test_loss, epoch)
    #     logger("test_accuracy", test_accuracy, epoch)
    #     summary_dict.save(checkpoint.dir / "summary_dict.npy")


def test_knn(
    model,
    device,
    train_loader,
    valid_loader,
    n_classes,
    logger,
    knn_n_neighbours=128,
    knn_reweighting_factor=0.1,
    knn_n_batches=512,
    knn_normalize_features=True,
    ):
    
    model.eval()
    
    # get train features for fitting knn
    train_features = []
    train_labels = []
    for step, (data, target) in enumerate(train_loader):
        data = data.to(device)
        _, embedding = model(data, return_embedding=False)
        if knn_normalize_features:
            embedding = F.normalize(embedding, dim=1)
        embedding = embedding.detach().cpu()
        train_features.append(embedding)
        
        target = target.detach().cpu()
        train_labels.append(target)
        
        if step == knn_n_batches:
            break
    
    train_features = torch.cat(train_features, dim=0).t().contiguous()
    logger("shape of knn train features", train_features.size())
    train_labels = torch.cat(train_labels, dim=0).t().contiguous()
    
    # knn predict
    test_features = []
    test_labels = []
    for step, (data, target) in enumerate(valid_loader):
        data = data.to(device)
        _, embedding = model(data, return_embedding=False)
        if knn_normalize_features:
            embedding = F.normalize(embedding, dim=1)
        embedding = embedding.detach().cpu()
        test_features.append(embedding)
        
        target = target.detach().cpu()
        test_labels.append(target)
        
        if step == knn_n_batches:
            break
    
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # print("test_features shape", test_features.size())
    # print("train_features shape", train_features.size())
    # print("train_labels shape", train_labels.size())
    pred_labels = knn_predict(test_features, train_features, train_labels, n_classes, knn_n_neighbours, knn_reweighting_factor)
    pred_labels = pred_labels.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    # print("pred_labels size", pred_labels.size())
    # print("test_labels size", test_labels.size())
    # print("test_labels view as", test_labels.view_as(pred_labels).size())
    correct = pred_labels.eq(test_labels.view_as(pred_labels)).sum().item()
    
    accuracy = 100. * correct / test_labels.size(0)
    logger("# of knn valid samples correct", correct)
    logger("knn valid accuracy", accuracy)
    return accuracy


def test_linear(config, byol_model: BYOL, device, train_loader, test_loader, local_rank):
    # todo
    byol_model.eval()
    model = byol_model.module.net
    model.fc = DDP(model.fc, device_ids=[local_rank], find_unused_parameters=True, )
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
        
        # logger(f"START epoch {epoch}")
        # logger.start_timer("train")
        
        train_loss = []
        for step, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            y = model(data)
            l = criterion(y, target)
            finetuning_optimizer.zero_grad()
            l.backward()
            finetuning_optimizer.step()
            train_loss.append(l.item())
            print('finetune train loss', l)
    
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
