## Bootstrap Your Own Latent (BYOL), in Pytorch

Practical implementation of an <a href="https://arxiv.org/abs/2006.07733">astoundingly simple method</a> for self-supervised learning that achieves a new state of the art (surpassing SimCLR) without contrastive learning and having to designate negative pairs.

This repository offers a module that one can easily wrap any neural network that accepts an image to immediately start benefitting from unlabelled data.

## Install

```bash
$ pip install byol-pytorch
```

## Usage

Simply plugin your neural network, specifying (1) the image dimensions as well as (2) the name (or index) of the hidden layer, whose output is used as the latent representation used for self-supervised training.

```python
import torch
from byol_pytorch import BYOL
from torchvision import models

resnet = models.resnet50(pretrained=True)

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')
```

That's pretty much it. After much training, the residual network should now perform better on its supervised downstream tasks.

## Advanced

While the hyperparameters have already been set to what the paper has found optimal, you can change them with extra keyword arguments to the base wrapper class.

```python
learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool',
    projection_size = 256,           # the projection size
    projection_hidden_size = 4096,	 # the hidden dimension of the MLP for both the projection and prediction
    moving_average_decay = 0.99      # the moving average decay factor for the target encoder, already set at what paper recommends
)
```

By default, this library will use the augmentations from the SimCLR paper, which is also used in the BYOL paper. However, if you would like to specify your own augmentations, you can simply pass in an `augment_fn` in the constructor. Augmentations must work in the tensor space. If you decide to use torchvision augmentations, make sure the function converts first to PIL `.toPILImage()` and then back to tensors `.ToTensor()`

```python
custom_augment_fn = nn.Sequential(
    kornia.augmentations.RandomHorizontalFlip()
)

learner = ContrastiveLearner(
    resnet,
    image_size = 256,
    hidden_layer = -2,
    augment_fn = custom_augment_fn
)
```

## Citation

```bibtex
@misc{grill2020bootstrap,
    title = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
    author = {Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
    year = {2020},
    eprint = {2006.07733},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
