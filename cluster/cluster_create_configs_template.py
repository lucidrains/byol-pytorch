import warnings

warnings.filterwarnings('ignore')
import argparse
import yaml

import copy
import os
import pickle
import pathlib
import numpy as np

parser = argparse.ArgumentParser(description='define cluster setup')
parser.add_argument('--project_dir', type=str, default=False, help='dir for project')
parser.add_argument('--gpus', type=int, default=False, help='number of GPUs')
parser.add_argument('--sess', type=str, default=False, help='session name')
args = parser.parse_args()

if args.project_dir:
    config_file = pathlib.Path(args.project_dir) / 'metassl' / 'default_config.yaml'
else:
    config_file = 'default_config.yaml'

with open(os.path.join("../metassl", config_file), 'r') as f:
    default_config = yaml.load(f, Loader=yaml.Loader)

if args.gpus:
    GPUS = args.gpus
else:
    GPUS = default_config['train']['gpus']

if args.sess:
    SESSION = args.sess
else:
    SESSION = "local_1"

PROJECT = "metassl"
RANDOM_SEEDS = 1

default_config['expt']['project_name'] = PROJECT
default_config['expt']['session_name'] = SESSION

np.random.seed(1)
seed_list = np.random.randint(1, 10000, RANDOM_SEEDS).tolist()

config_list = []
for seed in seed_list:
    for setting in ["grid", "random"]:
        
        copy_config = copy.deepcopy(default_config)
        copy_config['train']['gpus'] = GPUS
        copy_config['train']['seed'] = seed
        copy_config['model']['model_type'] = "resnet18"
        
        copy_config['data']['dataset'] = "CIFAR100"
        copy_config['train']['epochs'] = 200
        copy_config['train']['batch_size'] = 128
        copy_config['train']['warmup'] = 1000
        copy_config['train']['scheduler_epochs'] = 80
        
        if setting == "grid":
            
            for schedule in ["cosine", "cosineWarm", "step", "plateau", "const"]:
                for lr_high, in [
                    (0.1,),
                    (0.01,),
                    (0.001,),
                    (0.0001,),
                    ]:
                    if schedule == "const":
                        lr_low = lr_high
                        a_config = copy.deepcopy(copy_config)
                        a_config['optim']['schedule'] = schedule
                        a_config['optim']['lr_low'] = lr_low
                        a_config['optim']['lr_high'] = lr_high
                        
                        factor_list = [schedule, lr_high, lr_low]
                        factor_name = ["schedule", "lr_high", "lr_low"]
                        factor_str = []
                        for n, f in zip(factor_name, factor_list):
                            if isinstance(f, float):
                                f = int(f * 1000000)
                            factor_str.append(f"{n:.4}-{f}")
                        factor_str = "_".join(factor_str)
                        a_config['expt']['experiment_name'] = f"{setting}_{len(config_list)}_{factor_str}_seed-{seed}"
                        config_list.append(a_config)
                    else:
                        for lr_low, in [
                            (0.0001,),
                            (0.00001,),
                            (0.000001,),
                            (0.0000001,),
                            ]:
                            a_config = copy.deepcopy(copy_config)
                            a_config['optim']['schedule'] = schedule
                            a_config['optim']['lr_low'] = lr_low
                            a_config['optim']['lr_high'] = lr_high
                            
                            factor_list = [schedule, lr_high, lr_low]
                            factor_name = ["schedule", "lr_high", "lr_low"]
                            factor_str = []
                            for n, f in zip(factor_name, factor_list):
                                if isinstance(f, float):
                                    f = int(f * 1000000)
                                factor_str.append(f"{n:.4}-{f}")
                            factor_str = "_".join(factor_str)
                            a_config['expt']['experiment_name'] = f"{setting}_{len(config_list)}_{factor_str}_seed-{seed}"
                            config_list.append(a_config)
        
        
        
        elif setting == "random":
            RANDOM_SAMPLES = 68
            for schedule, lr_high in zip(
                np.random.choice(["cosine", "cosineWarm", "step", "plateau", "const"], RANDOM_SAMPLES),
                np.exp(np.random.uniform(np.log(0.1), np.log(0.0001), RANDOM_SAMPLES))
                ):
                
                if schedule == "const":
                    lr_low = lr_high
                else:
                    lr_low = np.exp(np.random.uniform(np.log(lr_high), np.log(0.0000001)))
                
                a_config = copy.deepcopy(copy_config)
                a_config['optim']['schedule'] = schedule
                a_config['optim']['lr_low'] = lr_low
                a_config['optim']['lr_high'] = lr_high
                
                factor_list = [schedule, lr_high, lr_low]
                factor_name = ["schedule", "lr_high", "lr_low"]
                factor_str = []
                for n, f in zip(factor_name, factor_list):
                    if isinstance(f, float):
                        f = int(f * 1000000)
                    factor_str.append(f"{n:.4}-{f}")
                factor_str = "_".join(factor_str)
                a_config['expt']['experiment_name'] = f"{setting}_{len(config_list)}_{factor_str}_seed-{seed}"
                config_list.append(a_config)

# check config
default_keys = sorted(list(default_config.keys()))
for config in config_list:
    config_keys = sorted(list(config.keys()))
    if len(config_keys) == len(default_keys):
        for ckey, dkey in zip(config_keys, default_keys):
            assert ckey == dkey
    else:
        raise UserWarning(f"missing config keys: {np.setdiff1d(default_keys, config_keys)}")

for idx, config in enumerate(config_list):
    config['experiment_idx'] = idx
    # print(idx, config['expt']['experiment_name'] )

if args.project_dir:
    with open(pathlib.Path(args.project_dir) / 'metassl' / f'cluster_configs_{SESSION}.pkl', 'wb') as f:
        pickle.dump(config_list, f)
else:
    with open(os.path.dirname(os.path.realpath(__file__)) + f'/cluster_configs_{SESSION}.pkl', 'wb') as f:
        pickle.dump(config_list, f)

print(len(config_list))
