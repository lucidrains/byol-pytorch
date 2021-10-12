from PIL import ImageFilter
import random
import torch
import torch.nn as nn
from torchvision.models import ResNet

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, num_classes=1000):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.backbone: ResNet = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = torch.nn.Identity()

        # build a 3-layer projector
        self.encoder_head = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                          nn.BatchNorm1d(prev_dim),
                                          nn.ReLU(inplace=True),  # first layer
                                          nn.Linear(prev_dim, prev_dim, bias=False),
                                          nn.BatchNorm1d(prev_dim),
                                          nn.ReLU(inplace=True),  # second layer
                                          nn.Linear(prev_dim, dim),
                                          nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder_head[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.classifier_head = nn.Linear(prev_dim, num_classes)

    def forward(self, x1, x2=None, finetuning=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        if finetuning:
            emb = self.backbone(x1)
            return self.classifier_head(emb)
        else:
            # compute features for one view
            emb1 = self.backbone(x1)  # NxC
            emb2 = self.backbone(x2)  # NxC

            z1 = self.encoder_head(emb1)
            z2 = self.encoder_head(emb2)

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            return p1, p2, z1.detach(), z2.detach()