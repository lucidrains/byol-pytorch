import os

import yaml


def get_expt_dir_with_bohb_config_id(expt_dir, bohb_config_id):
    config_id_path = "-".join(str(sub_id) for sub_id in bohb_config_id)
    expt_dir_id = os.path.join(expt_dir, config_id_path)
    return expt_dir_id


def organize_experiment_saving(user, config, is_bohb_run):
    # Set expt_root_dir based on user and experiment mode
    if user == "wagnerd":  # Diane cluster
        expt_root_dir = "/work/dlclarge2/wagnerd-metassl-experiments/metassl"
    else:
        expt_root_dir = "experiments"
    
    # Set expt_dir based on whether it is a BOHB run or not + differenciate between users
    if is_bohb_run:
        # for start_bohb_master (directory where config.json and results.json are being saved)
        expt_dir = os.path.join(expt_root_dir, "BOHB", config.data.dataset, config.expt.expt_name)
    else:
        if user == "wagn3rd" or user == "wagnerd":
            expt_dir = os.path.join(expt_root_dir, config.data.dataset, config.expt.expt_name)
        else:
            expt_dir = os.path.join(expt_root_dir, config.expt.expt_name)
    
    # TODO FABIO: if folder exists, create a new one with incremental suffix (-1, -2) and implement a flag ""
    # Create directory (if not yet existing) and save config.yaml
    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    
    with open(os.path.join(expt_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        print(f"copied config to {f.name}")
    
    return expt_dir


def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
