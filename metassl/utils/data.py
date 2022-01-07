import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler

from metassl.utils.imagenet import ImageNet
from .simsiam import GaussianBlur, TwoCropsTransform
from .torch_utils import DistributedSampler
import torchvision.datasets as datasets  # do not remove this

normalize_imagenet = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )

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
    parameterize_augmentation=False,
    bohb_infos=None,
    dataset_percentage_usage=100,
    use_fix_aug_params=False,
    data_augmentation_mode='default',
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

    if get_fine_tuning_loaders:
        print(f"using finetuning dataset: {dataset}")
    else:
        print(f"using pretraining dataset: {dataset}")

    if data_augmentation_mode == 'default':
        # default SimSiam Stuff + Fabio Stuff
        # TODO @Fabio/Diane - generate specific mode for Fabio stuff
        train_transform, valid_transform = get_train_valid_transforms(dataset_name, use_fix_aug_params, bohb_infos, get_fine_tuning_loaders, parameterize_augmentation)
    elif data_augmentation_mode == 'probability_augment':
        from .probability_augment import probability_augment
        train_transform, valid_transform = probability_augment(dataset_name, get_fine_tuning_loaders, bohb_infos, use_fix_aug_params)
    else:
        raise ValueError(f"Data augmentation mode {data_augmentation_mode} is not implemented yet!")

    if dataset_name == "ImageNet":
        # hardcoded for now
        root = "/data/datasets/ImageNet/imagenet-pytorch"
        # root = "/data/datasets/ILSVRC2012"

        # load the dataset
        train_dataset = ImageNet(
            root=root, split='train',
            transform=train_transform, ignore_archive=True,
            )
        
        valid_dataset = ImageNet(
            root=root, split='train',
            transform=valid_transform, ignore_archive=True,
            )
    elif dataset_name == "CIFAR10":
        print(f"{valid_size=}")
        if data_augmentation_mode == 'probability_augment' and not get_fine_tuning_loaders:
            from .albumentation_datasets import Cifar10Albumentations
            train_dataset = Cifar10Albumentations(root='datasets/CIFAR10', train=True,
                                                         download=True, transform=train_transform)
        else:
            train_dataset = torchvision.datasets.CIFAR10(root='datasets/CIFAR10', train=True,
                                                download=True, transform=train_transform)


        valid_dataset = torchvision.datasets.CIFAR10(root='datasets/CIFAR10', train=True,
                                               download=True, transform=valid_transform)
    else:
        # not supported
        raise ValueError('invalid dataset name=%s' % dataset)
    
    num_train = int(len(train_dataset) / 100 * dataset_percentage_usage)
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
        
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_imagenet,
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
    elif dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root='datasets/CIFAR10', train=False,
                                               download=True, transform=transform)
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


def get_train_valid_transforms(dataset_name, use_fix_aug_params, bohb_infos, get_fine_tuning_loaders, parameterize_augmentation):
    # ------------------------------------------------------------------------------------------------------------------
    # Specify data augmentation hyperparameters for the pretraining part
    # ------------------------------------------------------------------------------------------------------------------
    # TODO: @Diane - put that into a separate function
    # Defaults
    p_colorjitter = 0.8
    p_grayscale = 0.2
    p_gaussianblur = 0.5 if dataset_name == 'ImageNet' else 0
    brightness_strength = 0.4
    contrast_strength = 0.4
    saturation_strength = 0.4
    hue_strength = 0.1
    if use_fix_aug_params:
        # You can overwrite parameters here if you want to try out a specific setting.
        # Due to the flag, default experiments won't be affected by this.
        p_colorjitter = 0.8
        p_grayscale = 0.2
        p_gaussianblur = 0.5 if dataset_name == 'ImageNet' else 0
        brightness_strength = 1.1592547258007664
        contrast_strength = 1.160211615089221
        saturation_strength = 0.9843846879329252
        hue_strength = 0.19030216963226004

    # BOHB - probability augment configspace
    if bohb_infos is not None and bohb_infos['bohb_configspace'].endswith('probability_simsiam_augment'):
        p_colorjitter = bohb_infos['bohb_config']['p_colorjitter']
        p_grayscale = bohb_infos['bohb_config']['p_grayscale']
        p_gaussianblur =  bohb_infos['bohb_config']['p_gaussianblur'] if dataset_name == 'ImageNet' else 0

    # BOHB - color jitter strengths configspace
    if bohb_infos is not None and bohb_infos['bohb_configspace'] == 'color_jitter_strengths':
        brightness_strength = bohb_infos['bohb_config']['brightness_strength']
        contrast_strength = bohb_infos['bohb_config']['contrast_strength']
        saturation_strength = bohb_infos['bohb_config']['saturation_strength']
        hue_strength = bohb_infos['bohb_config']['hue_strength']

    # For testing
    # print(f"{p_colorjitter=}")
    # print(f"{p_grayscale=}")
    # print(f"{p_gaussianblur=}")
    # print(f"{brightness_strength=}")
    # print(f"{contrast_strength=}")
    # print(f"{saturation_strength=}")
    # print(f"{hue_strength=}")
    # ------------------------------------------------------------------------------------------------------------------

    if dataset_name == "CIFAR10":
        # No blur augmentation for CIFAR10!
        if not get_fine_tuning_loaders:
            if parameterize_augmentation:
                # rest is done outside
                # train_transform = transforms.Compose([
                #     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                #     transforms.ToTensor(),
                #     ])
                train_transform = None
            else:
                train_transform = TwoCropsTransform(
                    transforms.Compose(
                        [
                            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                            transforms.RandomApply([transforms.ColorJitter(brightness=brightness_strength, contrast=contrast_strength, saturation=saturation_strength, hue=hue_strength)], p=p_colorjitter),
                            transforms.RandomGrayscale(p=p_grayscale),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                            ]
                        )
                    )

            valid_transform = TwoCropsTransform(
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                    ]
                )
            )
        else:  # TODO: Check out which data augmentations are being used here!
            train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                ]
            )

            valid_transform = transforms.Compose([
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
        if not get_fine_tuning_loaders:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            if parameterize_augmentation:
                # rest is done outside
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.ToTensor(),
                    ])
            else:
                train_transform = TwoCropsTransform(
                    transforms.Compose(
                        [
                            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                            transforms.RandomApply([transforms.ColorJitter(brightness=brightness_strength, contrast=contrast_strength, saturation=saturation_strength, hue=hue_strength)], p=p_colorjitter),
                            transforms.RandomGrayscale(p=p_grayscale),
                            transforms.RandomApply([GaussianBlur([.1, 2.])], p=p_gaussianblur),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize_imagenet
                            ]
                        )
                    )

            valid_transform = TwoCropsTransform(
                transforms.Compose(
                    [
                        transforms.Resize(256, interpolation=Image.BICUBIC),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        normalize_imagenet
                        ]
                    )
                )
        else:
            train_transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize_imagenet,
                            ])
            # same as above without two crop transform
            valid_transform = transforms.Compose(
                    [
                        transforms.Resize(256, interpolation=Image.BICUBIC),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        normalize_imagenet
                        ]
                    )
    else:
        # not supported
        raise ValueError('invalid dataset name=%s' % dataset_name)

    return train_transform, valid_transform


def get_loaders(traindir, config, parameterize_augmentation=False, bohb_infos=None):
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
        parameterize_augmentation=parameterize_augmentation,
        bohb_infos=bohb_infos,
        dataset_percentage_usage=config.data.dataset_percentage_usage,
        use_fix_aug_params=config.expt.use_fix_aug_params,
        data_augmentation_mode=config.expt.data_augmentation_mode,
        )
    
    train_loader_ft, valid_loader_ft, train_sampler_ft, _ = get_train_valid_loader(
        data_dir=traindir,
        batch_size=config.finetuning.batch_size,
        random_seed=config.expt.seed,
        valid_size=config.finetuning.valid_size,
        dataset_name=config.data.dataset,
        shuffle=True,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=False,
        distributed=config.expt.distributed,
        drop_last=True,
        get_fine_tuning_loaders=True,
        parameterize_augmentation=False,
        bohb_infos=None,
        dataset_percentage_usage=config.data.dataset_percentage_usage,
        use_fix_aug_params=config.expt.use_fix_aug_params,
        data_augmentation_mode=config.expt.data_augmentation_mode,
        )
    
    test_loader_ft = get_test_loader(
        data_dir=traindir,
        batch_size=config.finetuning.batch_size,
        dataset_name=config.data.dataset,
        shuffle=False,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=False,
        drop_last=False,
        )
    
    if config.finetuning.valid_size > 0:
        return train_loader_pt, train_sampler_pt, train_loader_ft, train_sampler_ft, valid_loader_ft, test_loader_ft
    else:  # TODO: @Diane - Checkout and test on *parameterized_aug*
        return train_loader_pt, train_sampler_pt, train_loader_ft, train_sampler_ft, test_loader_ft

