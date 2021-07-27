import os, sys
from pathlib import Path
import numpy as np
import torch
import time
import datetime
import json

from metassl.utils.handler.base_handler import Handler
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from metassl.utils.handler.base_handler import Handler


class Logger(Handler):
    """
    call or .log() stores log massages in form (key, value, step) in json and prints them with timestamp.
    """
    
    def __init__(self, log_dir, file_name="log_file.txt", json_name="json_log.json"):
        super().__init__()
        
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / file_name
        self.json_file = self.log_dir / json_name
        
        self.timer_dict = {}
        self.line_dicts = []
        
        self.start_log()
    
    def __call__(self, key, value=None, time_step=None, rank=0):
        self.log(key, value, time_step, rank)
    
    def log(self, key, value=None, time_step=None, rank=0):
        if rank != 0:
            return
        
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        
        if isinstance(value, np.ndarray):
            value = value.tolist()
        
        if isinstance(time_step, torch.Tensor):
            time_step = time_step.cpu().detach().numpy()
        
        time_stamp = self.time_stamp()
        
        dump_dict = {
            "t": time_stamp
            }
        
        if value is None:
            if time_step is None:
                string = f"{time_stamp} {key}"
                dump_dict["k"] = str(key)
            else:
                string = f"{time_stamp} {key}  step:{time_step}"
                dump_dict["k"] = str(key)
                dump_dict["s"] = str(time_step)
        
        else:
            if isinstance(value, int):
                if value > 999:
                    value = f"{value:,}"
            if time_step is None:
                string = f"{time_stamp} {key}: {value}"
                dump_dict["k"] = str(key)
                dump_dict["v"] = str(value)
            else:
                string = f"{time_stamp} {key}: {value}  step: {time_step}"
                dump_dict["k"] = str(key)
                dump_dict["v"] = str(value)
                dump_dict["s"] = str(time_step)
        
        print(string)
        with open(self.log_file, 'a') as file:
            file.write(f"{string} \n")
        
        self.line_dicts.append(dump_dict)
    
    def start_log(self):
        if os.path.isfile(self.log_file) and os.access(self.log_file, os.R_OK):
            self.log("LOGGER: continue logging")
        else:
            with open(self.log_file, 'w+') as file:
                file.write(f"{self.time_stamp()} LOGGER: start logging with Python version: {str(sys.version).split('(')[0]} \n")
    
    def print_config(self, config, name="main"):
        if name == "main":
            self.log("#" * 20 + " CONFIG:")
        else:
            self.log(f"sub config {name:15}", np.unique([f"{attr}: {str(value)}" for attr, value in config.get_dict.items()]).tolist())
        
        if hasattr(config, "sub_config"):
            for cfg in config.sub_config:
                self.print_config(getattr(config, cfg), cfg)
    
    def start_timer(self, name, rank=0):
        name = f"{name}_{str(rank)}"
        self.timer_dict[name] = time.time()
    
    def timer(self, name, time_step=None, rank=0):
        name = f"{name}_{str(rank)}"
        if name not in self.timer_dict.keys():
            self.log("!!!!!! UNKNOWN TIMER", name, time_step)
        else:
            duration = time.time() - self.timer_dict[name]
            self.log(f"timer {name.split('_')[0]}", str(datetime.timedelta(seconds=duration)), time_step, rank)
    
    def save_to_json(self, rank=0):
        if rank != 0:
            return
        with open(self.json_file, 'w') as file:
            json.dump(self.line_dicts, file)
        self.log("LOGGER: save log to json")


if __name__ == "__main__":
    log_dir = "/rna_design/utils/log"
    
    log = Logger(log_dir)
    
    log.log("result", 123, 0)
    log.log("Hello World")
    
    log.log("result", 123, 1)
    
    log.dump(key="grads", value=[234.234, 234], time_step=0)
    
    log.log("result", 123, 2)
