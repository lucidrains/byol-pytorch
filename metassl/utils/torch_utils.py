import torch

def count_parameters(parameters):
    return sum(p.numel() for p in parameters if p.requires_grad)

