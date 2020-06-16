# Bootstrap Your Own Latent (BYOL), in Pytorch

Practical implementation of an astoundingly simple method for self-supervised learning that achieves a new state of the art (surpassing SimCLR) without contrastive learning and having to designate negative pairs. This repository offers a module that can wrap any neural network that accepts an image for this proposed form of self-supervised learning.

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
