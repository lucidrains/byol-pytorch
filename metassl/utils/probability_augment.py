
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2

from .simsiam import TwoCropsTransform

from albumentations import GaussNoise,ElasticTransform, \
    GridDistortion, OpticalDistortion, Normalize, RandomGridShuffle, RandomShadow, Blur, ColorJitter, Downscale, \
    Equalize, ChannelShuffle, GaussianBlur, GlassBlur, ImageCompression, InvertImg, ISONoise, JpegCompression, \
    MultiplicativeNoise
from albumentations.augmentations.geometric.transforms import Affine
from albumentations import Compose, OneOf


def probability_augment(
    dataset_name,
    get_fine_tuning_loaders,
    p_color_transformations=0.25,
    p_geometric_transformations=0.25,
    p_non_rigid_transformations=0.25,
    p_quality_transformations=0.25,
    p_exotic_transformations=0,
    augment_finetuning = False,
):
    # TODO: @Diane - Think about the seleted ops for: color, exotic, quality
    if dataset_name == "CIFAR10":
        if not get_fine_tuning_loaders:
            train_transform = Compose([
                    # color transformations
                    OneOf([
                        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1),
                        Equalize(p=1),
                        ChannelShuffle(p=1),
                        InvertImg(p=1)
                    ], p=p_color_transformations),
                    # geometric transformations
                    # TODO: @Diane - Checkout Affine
                    Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, always_apply=False, p=p_geometric_transformations),
                    # non-rigid transformations
                    OneOf([
                        ElasticTransform(p=1),
                        GridDistortion(p=1),
                        OpticalDistortion(p=1),
                    ], p=p_non_rigid_transformations),
                    # quality transformations
                    OneOf([
                        Blur(p=1),
                        Downscale(p=1),
                        GaussianBlur(p=1),
                        GaussNoise(p=1),
                        GlassBlur(p=1),
                        ImageCompression(p=1),
                        ISONoise(p=1),
                        JpegCompression(p=1),
                        MultiplicativeNoise(p=1)
                    ], p=p_quality_transformations),
                    # exotic transformations
                    OneOf([
                        RandomGridShuffle(p=1),
                        RandomShadow(p=1),
                    ], p=p_exotic_transformations),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    ToTensorV2(),
                ], p=1)

            valid_transform = TwoCropsTransform(
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                    ]
                )
            )
        else:
            p_augment_finetuning = 1 if augment_finetuning else 0  # TODO: @Diane - Implement that option
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
    return train_transform, valid_transform