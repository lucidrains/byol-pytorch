from typing import Optional, Callable, Tuple, Any

import torchvision
from PIL.Image import Image


class Cifar10Albumentations(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            image_a = self.transform(image=image)["image"]
            image_b = self.transform(image=image)["image"]
            image = [image_a, image_b]
        return image, label