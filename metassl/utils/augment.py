import math
from typing import Tuple, Dict

import torch
import torch as th
from torch import nn, Tensor
from torch.distributions.half_normal import HalfNormal
from torchvision.transforms import functional as F


class AugmentPerImage(nn.Module):
    def __init__(
        self, n_examples, num_magnitude_bins, std_aug_magnitude, extra_augs,
        same_across_batch
        ):
        super().__init__()
        self.same_across_batch = same_across_batch
        # This whole logic assumes the identity
        # is the smallest magnitude of augmentation
        if std_aug_magnitude is not None:
            dist = HalfNormal(num_magnitude_bins * std_aug_magnitude)
            cdfs = dist.cdf(th.linspace(1, num_magnitude_bins, num_magnitude_bins))
            # do not have to sum to zero as multinomial will normalize
            probs = th.cat((cdfs[0:1], th.diff(cdfs)))
        else:
            # Uniform sampling
            probs = torch.tensor([1 / num_magnitude_bins]).repeat(num_magnitude_bins)
        
        n_augs = 1 if same_across_batch else n_examples
        # loop over new variable n_augs instead of n_examples
        # if same across batch only n_augs =1, otherwise n_augs=n_examples
        self.op_names = []
        self.magnitudes = []
        for i_example in range(n_augs):
            op_meta = self._augmentation_space(
                num_magnitude_bins, extra_augs=extra_augs
                )
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            i_magnitude_bin = torch.multinomial(probs, 1).item()
            magnitude = (
                float(magnitudes[i_magnitude_bin].item())
                if magnitudes.ndim > 0
                else 0.0
            )
            if signed and torch.randint(2, (1,)).item():
                magnitude *= -1.0
            self.op_names.append(op_name)
            self.magnitudes.append(magnitude)
    
    def _augmentation_space(self, num_bins: int, extra_augs: bool) -> Dict[str, Tuple[Tensor, bool]]:
        space = {
            "Identity":   (torch.tensor(0.0), False),
            "ShearX":     (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY":     (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate":     (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color":      (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast":   (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness":  (torch.linspace(0.0, 0.99, num_bins), True),
            "Solarize":   (torch.linspace(256.0, 0.0, num_bins), False),
            }
        if extra_augs:
            extra_space = {
                "Posterize":    (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False,),
                "AutoContrast": (torch.tensor(0.0), False,),
                "Equalize":     (torch.tensor(0.0), False,),
                }
            space = {**space, **extra_space}
        return space
    
    def forward(self, X):
        # in case of same_across_batch just
        if self.same_across_batch:
            aug_X = _apply_op(X, self.op_names[0], self.magnitudes[0])
        # and keep final assert, skip everything else
        else:
            assert len(X) == len(self.op_names) == len(self.magnitudes)
            aug_Xs = []
            for i_image, (op_name, magnitude) in enumerate(
                zip(self.op_names, self.magnitudes)
                ):
                aug_X = _apply_op(X[i_image: i_image + 1], op_name, magnitude)
                aug_Xs.append(aug_X)
            aug_X = th.cat(aug_Xs)
        assert len(aug_X) == len(X)
        return aug_X


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    ):
    # from torchvision
    if op_name == "ShearX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(magnitude), 0.0],
            )
    elif op_name == "ShearY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(magnitude)],
            )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            shear=[0.0, 0.0],
            )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            shear=[0.0, 0.0],
            )
    elif op_name == "Rotate":
        img = F.rotate(
            img,
            magnitude,
            )
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = posterize_with_grad(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = autocontrast_with_grad(img)
    elif op_name == "Equalize":
        img = equalize_with_grad(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img
