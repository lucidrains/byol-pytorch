import argparse
import os
import pathlib
import pickle

from metassl.train_resnet import begin_training

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='define cluster setup')
    
    parser.add_argument('--job_idx', type=int, default=False, help='id of task')
    parser.add_argument('--job_name', type=str, default=False, help='name of task')
    parser.add_argument('--project_dir', type=str, default=False, help='dir for code')
    parser.add_argument('--data_dir', type=str, default=False, help='dir for data')
    parser.add_argument('--expt_dir', type=str, default=False, help='dir for expt')
    parser.add_argument('--sess', type=str, default=False, help='session name')
    
    args = parser.parse_args()
    
    if args.project_dir == False:
        print("no arguments in start experiment")
        project_dir = pathlib.Path(f"/home/{os.environ['USER']}/workspace/metassl/")
        expt_dir = pathlib.Path(f"/home/{os.environ['USER']}/workspace/experiments/tmp/")
        job_idx = 0
        job_name = "local_run"
        data_dir = f"/home/{os.environ['USER']}/workspace/data/metassl"
        session = "local_1"
    else:
        job_idx = args.job_idx
        job_name = args.job_name
        session = args.sess
        project_dir = pathlib.Path(args.project_dir)
        expt_dir = pathlib.Path(args.expt_dir)
        data_dir = pathlib.Path(args.data_dir)
    
    print("job_idx ", job_idx)
    print("job_name ", job_name)
    print("project_dir ", project_dir)
    print("expt_dir ", expt_dir)
    print("data_dir ", data_dir)
    
    with open(project_dir / 'metassl' / f'cluster_configs_{session}.pkl', 'rb') as f:
        unpickler = pickle.Unpickler(f)
        config_list = unpickler.load()
    
    config = config_list[int(job_idx)]
    
    config['expt']['job_name'] = job_name
    
    print("config: ", config)
    print("start_cluster_training")
    
    os.chdir(project_dir / "boho")
    begin_training(config, expt_dir)
