import os
import pathlib
import time

import numpy as np
import torch

"""
save and restore checkpoints including parameters, rng states and env/data states
"""


class CheckpointHandler():
    
    def __init__(self, checkpoint_dir, ):
        self.dir = pathlib.Path(checkpoint_dir)
    
    def save_training(self, mode_state_dict, optimizer_state_dict, epoch=None, loss=None, number=0):
        torch.save(
            {
                'epoch':                epoch,
                'model_state_dict':     mode_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'loss':                 loss,
                }, self.dir / f"training_{number}.tar"
            )
    
    def load_training(self, number=0):
        checkpoint = torch.load(self.dir / f"training_{number}.tar")
        mode_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return mode_state_dict, optimizer_state_dict, epoch, loss
    
    def save_model(self, model, number=0):
        torch.save(model, self.dir / f"model_{number}.pth")
    
    def save_optimizer(self, optimizer, number=0):
        torch.save(optimizer.optimizer, self.dir / f"optimizer_{number}.pth")
    
    def load_newest_optimizer(self, optimizer, map_location=None):
        newest_file = ""
        newest_age = 1e24
        for file in os.listdir(self.dir):
            if file.endswith(".pth") and "optimizer" in file.__str__():
                file_stat = os.stat(self.dir / file)
                file_age = (time.time() - file_stat.st_mtime)
                if file_age < newest_age:
                    newest_file = file
                    newest_age = file_age
        print("load optimizer ", newest_file)
        optimizer.optimizer = torch.load(self.dir / newest_file, map_location=map_location)
    
    def load_optimizer(self, optimizer, number=0, map_location=None):
        optimizer.optimizer = torch.load(self.dir / f"optimizer_{number}.pth", map_location=map_location)
    
    def load_model(self, number=0, map_location=None):
        model = torch.load(self.dir / f"model_{number}.pth", map_location=map_location)
        return model
    
    def load_newest_model(self, map_location=None):
        newest_file = ""
        newest_age = 1e24
        for file in os.listdir(self.dir):
            if file.endswith(".pth") and "model" in file.__str__():
                file_stat = os.stat(self.dir / file)
                file_age = (time.time() - file_stat.st_mtime)
                if file_age < newest_age:
                    newest_file = file
                    newest_age = file_age
        print("load file ", newest_file)
        model = torch.load(self.dir / newest_file, map_location=map_location)
        return model
    
    def model_exists(self):
        return os.path.isfile(self.dir / f"model_0.pth")
    
    def save_state_dict(self, state_dict, number=0):
        torch.save(state_dict, self.dir / f"state_dict_{number}.pth")
    
    def load_state_dict(self, number=0, cpu=True):
        if cpu:
            state_dict = torch.load(self.dir / f"state_dict_{number}.pth", map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(self.dir / f"state_dict_{number}.pth")
        return state_dict
    
    def save_object(self, object, name="object_0"):
        np.save(self.dir / f"{name}.npy", object, allow_pickle=True)
    
    def load_object(self, name="object_0"):
        return np.load(self.dir / f"{name}.npy", allow_pickle=True)
