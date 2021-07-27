import warnings

warnings.filterwarnings('ignore')
import os
import pathlib
import random
import yaml

import numpy as np
import torch
from torch import nn

# original torchvision resnet implementation:
from torchvision.models import resnet18, resnet50  # torchvision (for ImageNet)

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


def train_model(config, logger, checkpoint):
    logger.print_config(config)
    
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, valid_loader = get_train_valid_loader(
        data_dir=config.data.data_dir,
        batch_size=config.train.batch_size,
        random_seed=config.data.seed,
        dataset_name=config.data.dataset,
        num_workers=50,
        pin_memory=False,
        download=True,
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
        model = resnet18(num_classes=out_size).to(device)
        # model = resnet20(num_classes=out_size).to(device)
    elif config.model.model_type == "resnet50":
        model = resnet50(num_classes=out_size).to(device)
        # model = resnet56(num_classes=out_size).to(device)
    
    if torch.cuda.device_count() > 1 and config.expt.ddp:
        logger.log("Using DDP with # GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)
        model.to(device)
    
    logger.log("model_parameters", count_parameters(model.parameters()))
    
    summary_dict = SummaryDict()
    
    logger("START initial validation")
    
    valid_loss, accuracy = test(model, device, valid_loader)
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
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    for epoch in range(config.train.epochs):
        
        logger(f"START epoch {epoch}")
        logger.start_timer("train")
        
        model.train()
        train_loss = []
        for step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step(step, valid_loss)
            train_loss.append(loss.detach().cpu().numpy())
            
            if step % config.train.eval_freq == 0:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                print(
                    f"Epoch {epoch:{5}} / {config.train.epochs:{5}}, Accuracy {100 * correct / config.train.batch_size:{2}.2f}%, Loss"
                    f" {loss.item():.5f}"
                    )
        
        if config.expt.save_model and epoch % 5 == 0:
            checkpoint.save_training(
                mode_state_dict=model.state_dict(),
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
    test_loss, test_accuracy = test(model, device, test_loader)
    summary_dict["test_loss"] = test_loss
    summary_dict["test_accuracy"] = test_accuracy
    logger("test_loss", test_loss, epoch)
    logger("test_accuracy", test_accuracy, epoch)
    
    summary_dict.save(checkpoint.dir / "summary_dict.npy")


def test(model, device, test_loader):
    model.eval()
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
    
    sup = Supporter(experiments_dir=expt_dir, config_dict=config, count_expt=True, reload_expt=reload_expt)
    config = sup.get_config()
    logger = sup.get_logger()
    
    train_model(config=config, logger=logger, checkpoint=sup.ckp)


if __name__ == "__main__":
    
    user = os.environ.get('USER')
    
    with open("metassl/default_config.yaml", "r") as f:
        config = yaml.load(f)
    
    expt_dir = f"/home/{user}/workspace/experiments/metassl"
    config['data']['data_dir'] = f'/home/{user}/workspace/data/metassl'
    
    begin_training(config=config, expt_dir=expt_dir)
