from typing import List
import os, sys
import pathlib
from pathlib import Path
from metassl.utils.handler.config import ConfigHandler

def find_experiments(expt_dir, search_keys):


    expt_dir = Path(expt_dir)
    expt_list = []

    for subdir, dirs, files in os.walk(expt_dir):
        subdir = Path(subdir)

        if os.path.exists(subdir / 'summary_dict.npy') and os.path.exists(subdir / 'config.yml') and all([key in subdir.__str__() for key in search_keys]):
            config = ConfigHandler(config_file=subdir / 'config.yml')
            # expt_name = config.expt.session_name + "/" + config.expt.experiment_name.split("_seed")[0]
            expt_name = config.expt.session_name + "/" + config.expt.experiment_name
            seed = config.train.seed
            expt_dict = {"dir": subdir, "expt_name":expt_name, "seed":seed, "expt_folder": config.expt.experiment_name }

            expt_list.append(expt_dict)
    return expt_list


if __name__ == "__main__":

    user = os.environ.get('USER')
    expt_dir = f"/home/{user}/workspace/experiments/boho"

    search_key_list = [
        ['rna', "_32_", "opti_1", "2576"],
        ['rna', "_32_", "opti_1", "7337"],
        ['rna', "_32_", "opti_2"],
        ['rna', "_33_", "all"],
    ]



    list_expt = find_experiments(expt_dir, search_key_list[0])

    print(list_expt)