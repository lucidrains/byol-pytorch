import glob
import io
import math
import os
import shutil
from typing import TypeVar, Optional, Iterator

import PIL.Image
import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from torch.utils.data import Sampler
from torchvision.transforms import ToTensor


def count_parameters(parameters):
    return sum(p.numel() for p in parameters if p.requires_grad)


T_co = TypeVar('T_co', covariant=True)


class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """
    
    def __init__(
        self, indices: torch.Tensor, num_replicas: Optional[int] = None,
        rank: Optional[int] = None, shuffle: bool = True,
        seed: int = 0, drop_last: bool = False
        ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.indices = indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.indices) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.indices) - self.num_replicas) / self.num_replicas  # type: ignore
                )
        else:
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
    
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.indices), generator=g).tolist()  # type: ignore
            indices = self.indices[torch.randperm(len(self.indices), generator=g)].tolist()  # type: ignore
        else:
            indices = list(self.indices)  # type: ignore
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def knn_predict(
    feature, feature_bank, feature_labels, classes: int, knn_k: int, knn_t: float
    ):
    """Helper method to run kNN predictions on features based on a feature bank
    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t:
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
        )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
        )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
        )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def get_newest_model(path, suffix="*.pth.tar"):
    file_list = glob.glob(os.path.join(path, suffix))
    file_list = sorted(file_list, key=lambda x: x[:-4])
    if file_list:
        return file_list[-1]


def deactivate_bn(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm1d) or isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, torch.nn.Identity())
        else:
            deactivate_bn(child)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def check_and_save_checkpoint(config, ngpus_per_node, total_iter, epoch, model, optimizer_pt, optimizer_ft, expt_dir, optimizer_aug=None):
    if not config.expt.multiprocessing_distributed or (config.expt.multiprocessing_distributed and config.expt.rank % ngpus_per_node == 0):
        if epoch % config.expt.save_model_frequency == 0:
            save_dct = {
                'total_iter':   total_iter,
                'epoch':        epoch + 1,
                'arch':         config.model.model_type,
                'state_dict':   model.state_dict(),
                'optimizer_pt': optimizer_pt.state_dict(),
                'optimizer_ft': optimizer_ft.state_dict(),
                }
            if optimizer_aug is not None:
                save_dct['optimizer_aug'] = optimizer_aug.state_dict()
            
            save_checkpoint(save_dct, is_best=False, filename=os.path.join(expt_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch)))


def hist_to_image(hist_dict, title=None):
    plt.figure()
    plt.bar(hist_dict.keys(), hist_dict.values(), width=0.05)
    if title:
        plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image
