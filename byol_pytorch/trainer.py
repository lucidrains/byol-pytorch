from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn import SyncBatchNorm

from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader

from byol_pytorch.byol_pytorch import BYOL

from beartype import beartype
from beartype.typing import Optional

from accelerate import Accelerator

# functions

def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# class

class MockDataset(Dataset):
    def __init__(self, image_size, length):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(3, self.image_size, self.image_size)

# main trainer

class BYOLTrainer(Module):
    @beartype
    def __init__(
        self,
        net: Module,
        *,
        image_size: int,
        hidden_layer: str,
        learning_rate: float,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int = 16,
        optimizer_klass = Adam,
        checkpoint_every: int = 1000,
        checkpoint_folder: str = './checkpoints',
        byol_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerator_kwargs)

        if dist.is_initialized() and dist.get_world_size() > 1:
            net = SyncBatchNorm.convert_sync_batchnorm(net)

        self.net = net

        self.byol = BYOL(net, image_size = image_size, hidden_layer = hidden_layer, **byol_kwargs)

        self.optimizer = optimizer_klass(self.byol.parameters(), lr = learning_rate, **optimizer_kwargs)

        self.dataloader = DataLoader(dataset, shuffle = True, batch_size = batch_size)

        self.num_train_steps = num_train_steps

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        # prepare with accelerate

        (
            self.byol,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.byol,
            self.optimizer,
            self.dataloader
        )

        self.register_buffer('step', torch.tensor(0))

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def forward(self):
        step = self.step.item()
        data_it = cycle(self.dataloader)

        for _ in range(self.num_train_steps):
            images = next(data_it)

            with self.accelerator.autocast():
                loss = self.byol(images)
                self.accelerator.backward(loss)

            self.print(f'loss {loss.item():.3f}')

            self.optimizer.zero_grad()
            self.optimizer.step()

            self.wait()

            self.byol.update_moving_average()

            self.wait()

            if not (step % self.checkpoint_every) and self.accelerator.is_main_process:
                checkpoint_num = step // self.checkpoint_every
                checkpoint_path = self.checkpoint_folder / f'checkpoint.{checkpoint_num}.pt'
                torch.save(self.net.state_dict(), str(checkpoint_path))

            self.wait()

            step += 1

        self.print('training complete')
