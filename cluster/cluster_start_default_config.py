import argparse
import os
import pathlib

import yaml

from metassl.train_resnet import begin_training

"""

python cluster_start_default_config.py --data_dir="/home/ferreira/workspace/data/metassl"
--expt_dir="/home/ferreira/workspace/experiments/" --project_dir="/home/ferreira/workspace/metassl"

"""

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='define cluster setup')
    
    parser.add_argument('--project_dir', type=str, default=False, help='dir for code')
    parser.add_argument('--data_dir', type=str, default=False, help='dir for data')
    parser.add_argument('--expt_dir', type=str, default=False, help='dir for expt')
    parser.add_argument('--sess', type=str, default=False, help='dir for expt')
    parser.add_argument('--job_name', type=str, default=False, help='name of task')
    
    args = parser.parse_args()
    
    if args.project_dir == False:
        print("no arguments in start experiment")
        project_dir = pathlib.Path(f"/home/{os.environ['USER']}/workspace/metassl/")
        expt_dir = pathlib.Path(f"/home/{os.environ['USER']}/workspace/experiments/")
        data_dir = f"/home/{os.environ['USER']}/workspace/data/metassl"
        session = "local_1"
    else:
        project_dir = pathlib.Path(args.project_dir)
        expt_dir = pathlib.Path(args.expt_dir)
        data_dir = pathlib.Path(args.data_dir)
        session = args.sess
    
    print("project_dir ", project_dir)
    print("expt_dir ", expt_dir)
    print("data_dir ", data_dir)
    

    config_file = pathlib.Path(args.project_dir) / 'metassl' / 'default_config.yaml'
    
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    print(args.job_name)
    if args.job_name == False:
        job_name = "default_config"
    else:
        job_name = args.job_name
        
    config['expt']['job_name'] = job_name
    config['expt']['session_name'] = session
    config['expt']['experiment_name'] = "default_config"
    config['data']['data_dir'] = data_dir
    
    print("config: ", config)
    print("start_cluster_training")
    
    os.chdir(project_dir / "metassl")
    begin_training(config, expt_dir)
