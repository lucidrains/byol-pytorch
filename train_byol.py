import torch
from torchvision import models

from byol_pytorch import BYOL

resnet = models.resnet50(pretrained=True)

learner = BYOL(
        resnet,
        image_size=256,
        hidden_layer='avgpool'
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
    learner.update_moving_average()  # update moving average of target encoder

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')
