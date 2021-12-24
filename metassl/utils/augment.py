import torch
import torchvision.transforms
import torchvision.transforms as transforms

from metassl.utils.data import normalize_imagenet
from metassl.utils.simsiam import TwoCropsTransform


def create_transforms(strengths_b, strengths_c, strengths_s, strengths_h, image_height=224, image_width=224):
    if isinstance(strengths_b, float):
        strengths_b, strengths_c, strengths_s, strengths_h = [strengths_b], [strengths_c], [strengths_s], [strengths_h]
    
    assert len(strengths_b) == len(strengths_c) == len(strengths_s) == len(strengths_h), "strengths must all have the same length"
    
    transforms_list = []
    
    kernel_h = image_height // 10
    kernel_w = image_width // 10
    
    if kernel_h % 2 == 0:
        kernel_h -= 1
    if kernel_w % 2 == 0:
        kernel_w -= 1
    
    for i in range(len(strengths_b)):
        
        trans = TwoCropsTransform(
            transforms.Compose(
                [
                    transforms.RandomApply([transforms.ColorJitter(brightness=strengths_b[i], contrast=strengths_c[i], saturation=strengths_s[i], hue=strengths_h[i])], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([torchvision.transforms.GaussianBlur([kernel_h, kernel_w], [.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    normalize_imagenet
                    ]
                )
            )
        
        transforms_list.append(trans)
    
    return transforms_list


def augment_per_image(transformed_list, images):
    assert len(transformed_list) == images.shape[0]
    
    print(images.size())
    images = [transform(img.unsqueeze(0)) for transform, img in zip(transformed_list, images)]
    print(images)
    print(torch.cat(images, dim=1).size())
    
    return torch.cat(images, dim=1)
