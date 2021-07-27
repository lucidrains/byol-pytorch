import os
import pathlib
import random

import numpy as np
import torch
from torch import nn

from metassl.utils.my_optimizer import MyOptimizer
from metassl.utils.summary import SummaryDict
from metassl.utils.supporter import Supporter
from metassl.utils.torch_utils import count_parameters
from metassl.data.image_utils import get_test_loader, get_train_valid_loader

# original torchvision resnet implementation:
from torchvision.models import resnet18, resnet50  # torchvision (for ImageNet)

# custom implementation; faster but smaller complexity through fewer conv layers and smaller fc layer (for CIFAR10/100):
# from metassl.models.ResNet_CIFAR_small import resnet20, resnet56

# custom implementation; close to the original implementation with 3x3 instead of 7x7 conv1 and stride 1 (for CIFAR10/100):
# from metassl.models.ResNet_CIFAR import resnet18, resnet50

print("CUDA", torch.cuda.is_available())


def train_model(config, logger, checkpoint):
    cfg = config
    log = logger
    ckp = checkpoint
    log.print_config(cfg)

    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.train.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # TODO ImageNet case: pass cfg.data into below method and manually set ImageNet path("/data/datasets/ImageNet/imagenet-pytorch")
    train_loader, valid_loader = get_train_valid_loader(data_dir=cfg.data.data_dir,
                                                        batch_size=cfg.train.batch_size,
                                                        random_seed=cfg.data.seed,
                                                        augment=cfg.data.augment,
                                                        dataset_name=cfg.data.dataset,
                                                        num_workers=1,
                                                        pin_memory=False,
                                                        )
    test_loader = get_test_loader(data_dir=cfg.data.data_dir,
                                  batch_size=cfg.train.batch_size,
                                  shuffle=False,
                                  dataset_name=cfg.data.dataset,
                                  num_workers=1,
                                  pin_memory=False,
                                  )

    out_size = len(train_loader.dataset.classes)

    log("device", device)
    log("train dataset shape", train_loader.dataset.data.shape)
    log("out_size", out_size)

    if cfg.model.model_type == "resnet18":
        model = resnet18(num_classes=out_size).to(device)
        # model = resnet20(num_classes=out_size).to(device)
    elif cfg.model.model_type == "resnet50":
        model = resnet50(num_classes=out_size).to(device)
        # model = resnet56(num_classes=out_size).to(device)

    log.log("model_parameters", count_parameters(model.parameters()))

    summary_dict = SummaryDict()

    log("START initial validation")

    valid_loss, accuracy = test(model, device, valid_loader)
    summary_dict["step"] = 0
    summary_dict["valid_loss"] = valid_loss
    summary_dict["valid_accuracy"] = accuracy
    summary_dict["train_loss"] = 0
    summary_dict["learning_rate"] = 0

    iter_per_epoch = len(train_loader)
    max_steps = cfg.train.epochs * iter_per_epoch
    log("iter_per_epoch", iter_per_epoch)
    log("max_steps", max_steps)

    # TODO model dim:
    optimizer = MyOptimizer(0, model.parameters(), max_steps, iter_per_epoch, **cfg.optim.get_dict)

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(cfg.train.epochs):

        log(f"START epoch {epoch}")
        log.start_timer("train")

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

            if step % 100 == 0:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                print(f"Epoch {epoch:{5}} / {cfg.train.epochs:{5}}, Accuracy {100 * correct / cfg.train.batch_size:{2}.2f}%, Loss"
                      f" {loss.item():.5f}")

        log.timer("train", epoch)

        valid_loss, accuracy = test(model, device, valid_loader)
        summary_dict["step"] = epoch + 1
        summary_dict["train_loss"] = np.mean(train_loss)
        summary_dict["valid_loss"] = valid_loss
        summary_dict["valid_accuracy"] = accuracy
        summary_dict["learning_rate"] = optimizer._rate
        log("train_loss", np.mean(train_loss), epoch)
        log("valid_loss", valid_loss, epoch)
        log("valid_accuracy", accuracy, epoch)
        log("learning_rate", optimizer._rate, epoch)

        summary_dict.save(ckp.dir / "summary_dict.npy")

    # TEST FINAL MODEL
    test_loss, test_accuracy = test(model, device, test_loader)
    summary_dict["test_loss"] = test_loss
    summary_dict["test_accuracy"] = test_accuracy
    log("test_loss", test_loss, epoch)
    log("test_accuracy", test_accuracy, epoch)

    summary_dict.save(ckp.dir / "summary_dict.npy")


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


def create_support_training(config, expt_dir):
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

    config = {
            "expt": {
                    "project_name": "boho",
                    "session_name": "development",
                    "experiment_name": "local_2",
                    "job_name": "local_run",
                    "save_model": False,
                    "resume_training": False,
                    "resume_optimizer": False,
                    },
            "train": {
                    # "eval_freq":        5,
                    "seed": 123,
                    "batch_size": 128,
                    "epochs": 200,
                    },
            "criterion": {
                    # "loss":             "cross_entropy",  # geco elbo cross_entropy label_smoothing bce sce bf1 sf1

                    },
            "model": {
                    "model_type": "resnet18",  # resnet18, resnet50,
                    "seed": 123,

                    },
            "optim": {
                    "optimizer": "adam",  # adam adamW rmsprop adabelief sgd
                    "schedule": "const",  # noam cosine cosineWarm plateau step exponential const (set lr_low==lr_high)
                    "warmup": 1000,  # 0 (turned off) or higher (e.g. 1000 ~ 5 epochs at batch size 256 on CIFAR100)
                    "factor": 0.1,  # 1.0
                    "lr_low": 0.1,
                    "lr_high": 0.1,
                    "clip_grad": False,
                    "weight_decay": 0.0001,
                    "scheduler_epochs": 80,  # after how many epochs should the scheduler execute a step, only used in step, cosine,
                    # cosineWarm
                    },
            "data": {
                    "seed": 123,
                    "data_dir": f'/home/{user}/workspace/data/boho',
                    "dataset": "CIFAR100",  # CIFAR10, CIFAR100, ImageNet
                    "augment": True,
                    },

            }

    expt_dir = f"/home/{user}/workspace/experiments/boho"
    config['data']['data_dir'] = f'/home/{user}/workspace/data/boho'

    print("TRAIN LOCAL")
    create_support_training(config=config, expt_dir=expt_dir)
