import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler

from metassl.utils.imagenet import ImageNet
from .simsiam import GaussianBlur, TwoCropsTransform
from .torch_utils import DistributedSampler
import torchvision.datasets as datasets  # do not remove this

def get_train_valid_loader(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=1,
    pin_memory=True,
    download=True,
    dataset_name="ImageNet",
    distributed=False,
    drop_last=True,
    get_fine_tuning_loaders=False,
    ):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over a dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - dataset_name: the dataset name as a string, supported: "CIFAR10", "CIFAR100", "ImageNet"
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    # error_msg = "[!] valid_size should be in the range [0, 1]."
    # assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    allowed_datasets = ["CIFAR10", "CIFAR100", "ImageNet"]
    if dataset_name not in allowed_datasets:
        print(f"dataset name should be in {allowed_datasets}")
        exit()
    
    dataset = eval("datasets." + dataset_name)
    print(f"using dataset: {dataset}")
    
    if dataset_name == "CIFAR10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                ]
            )
        
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                
                ]
            )
    
    elif dataset_name == "CIFAR100":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
                ]
            )
        
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
                ]
            )
    
    elif dataset_name == "ImageNet":
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )

        if not get_fine_tuning_loaders:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            train_transform = TwoCropsTransform(
                transforms.Compose(
                    [
                        
                        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                        transforms.RandomApply(
                            [
                                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                                ], p=0.8
                            ),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize
                        ]
                    )
                )
                
            valid_transform = TwoCropsTransform(
                transforms.Compose(
                    [
                        transforms.Resize(256, interpolation=Image.BICUBIC),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        normalize
                        ]
                    )
                )
        else:
            train_transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ])
            # same as above without two crop transform
            valid_transform = transforms.Compose(
                    [
                        transforms.Resize(256, interpolation=Image.BICUBIC),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        normalize
                        ]
                    )
    else:
        # not supported
        raise ValueError('invalid dataset name=%s' % dataset)

    if dataset_name == "ImageNet":
        # hardcoded for now
        root = "/data/datasets/ImageNet/imagenet-pytorch"
        # root = "/data/datasets/ILSVRC2012"
        # print("---------------------using new ImageNet----------------------")
        # root = "/data/datasets/ImageNet/imagenet-pytorch/imagenet-pytorch-2-dont-use"
        # load the dataset
        train_dataset = ImageNet(
            root=root, split='train',
            transform=train_transform, ignore_archive=True,
            )
        
        valid_dataset = ImageNet(
            root=root, split='train',
            transform=valid_transform, ignore_archive=True,
            )
    else:
        # load the dataset
        train_dataset = dataset(
            root=data_dir, train=True,
            download=download, transform=train_transform,
            )
        
        valid_dataset = dataset(
            root=data_dir, train=True,
            download=download, transform=valid_transform,
            )
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    if np.isclose(valid_size, 0.0):
        train_idx, valid_idx = indices, indices
    else:
        train_idx, valid_idx = indices[split:], indices[:split]
    
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    if distributed:
        train_sampler = DistributedSampler(torch.tensor(train_idx))
        # TODO: use distributed valid_sampler and average accuracies to make validation more efficient
    else:
        train_sampler = SubsetRandomSampler(train_idx)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
        )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
        )
    
    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)
        
    if np.isclose(valid_size, 0.0):
        return train_loader, None, train_sampler, None
    else:
        return train_loader, valid_loader, train_sampler, valid_sampler


def get_test_loader(
    data_dir,
    batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    download=True,
    dataset_name="ImageNet",
    drop_last=False,
    ):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-100 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - dataset_name: the dataset name as a string, supported: "CIFAR10", "CIFAR100", "ImageNet"
    Returns
    -------
    - data_loader: test set iterator.
    """
    allowed_datasets = ["CIFAR10", "CIFAR100", "ImageNet"]
    if dataset_name not in allowed_datasets:
        print(f"dataset name should be in {allowed_datasets}")
        exit()
    
    dataset = eval("datasets." + dataset_name)
    
    if dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                ]
            )
    
    elif dataset_name == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
                ]
            )
    
    elif dataset_name == "ImageNet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ]
            )
    
    else:
        # not supported
        raise ValueError('invalid dataset name=%s' % dataset)
    
    if dataset_name == "ImageNet":
        # hardcoded for now
        # TODO: move to imagenet.py
        root = "/data/datasets/ImageNet/imagenet-pytorch"
        # load the dataset
        dataset = ImageNet(
            root=root, split="val",
            transform=transform, ignore_archive=True,
            )
    else:
        # load the dataset
        dataset = dataset(
            root=data_dir, train=False,
            download=download, transform=transform,
            )
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last,
        )
    
    return data_loader


def plot_images(images, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    label_names = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
        ]
    
    fig, axes = plt.subplots(3, 3)
    
    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation='spline16')
        
        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
                )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
