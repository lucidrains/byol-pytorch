from tqdm import tqdm
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler


try:
    from metassl.utils.data import normalize_cifar10, normalize_imagenet, normalize_cifar100
    from metassl.utils.imagenet import ImageNet
except ImportError:
    from .utils.data import normalize_cifar10, normalize_imagenet
    from .utils.imagenet import ImageNet


def get_knn_data_loaders(batch_size, num_workers, dataset):
    """
    Data loader for kNN classifier.
    Needs to be the training data, test data, with test augmentation and no shuffling
    @param batch_size
    @param num_workers
    @param dataset Cifar10/ImageNet/Cifar100
    """
    if dataset == "CIFAR10":
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize_cifar10])

        memory_data = torchvision.datasets.CIFAR10(root='datasets/CIFAR10', train=True,
                                                   transform=test_transform, download=True)
        test_data = torchvision.datasets.CIFAR10(root='datasets/CIFAR10', train=False,
                                                 transform=test_transform, download=True)
    elif dataset== "ImageNet":

        # taken from get_test_loader in data.py
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_imagenet,
            ]
        )
        # taken from get_train_valid_loader in data.py
        # hardcoded for now
        root = "/data/datasets/ImageNet/imagenet-pytorch"
        # root = "/data/datasets/ILSVRC2012"
        # load the dataset
        memory_data = ImageNet(root=root, split='train',
                                 transform=test_transform,
                                 ignore_archive=True,
        )
        # load the dataset
        test_data = ImageNet(
            root=root, split="val",
            transform=test_transform,
            ignore_archive=True,
        )
    elif dataset == "CIFAR100":
        # adding for the sake of completeness
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize_cifar100
            ]
        )
        memory_data = torchvision.datasets.CIFAR100(root='datasets/CIFAR100', train=True,
                                                   transform=test_transform, download=True)
        test_data = torchvision.datasets.CIFAR100(root='datasets/CIFAR100', train=False,
                                                 transform=test_transform, download=True)
    else:
        # not supported
        raise ValueError('invalid dataset name=%s' % dataset)

    memory_loader = torch.utils.data.DataLoader(memory_data, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)

    return memory_loader, test_loader


# Code from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
# test using a knn monitor
def knn_classifier(net, batch_size, workers, dataset, k=200, t=0.1, hide_progress=False):
    # Moco used 200
    """
     @param net: Model backbone. Encoder in our case
     @workers
     @param dataset: ImageNet CIFAR10 CIFAR100
     @param k: top neighbors to find. 200 is for ImageNet
    """
    # separate loaders used since the training and validation loaders used during pretraining are shuffled.
    memory_data_loader, test_data_loader = get_knn_data_loaders(batch_size=batch_size, num_workers=workers, dataset=dataset)
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy': total_top1 / total_num * 100})
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
