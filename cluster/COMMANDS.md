Remote Deployment
1) create Deployment
- PyCharm Preferences -> Build, Execution, Deployment -> Deployment -> click on "+" -> click on SFTP, set name 
  -> create or select existing SSH config 
  -> Mappings -> set Deployment path (location of project on cluster)
2) set Deployment in Python Interpreter
- PyCharm Preferences -> go to Project: metassl -> Python Interpreter ->  click on wheel -> add -> SSH interpreter 
  -> select previously created SSH config -> click next -> set remote python path (the one from your conda env) 
3) when running .py file, choose the Deployment Python Interpreter


/home/frankej/miniconda3/envs/metassl/bin/python3

get IP address: ifconfig -a


https://aadwiki.informatik.uni-freiburg.de/Meta_Slurm

ssh frankej@aadlogin.informatik.uni-freiburg.de

ssh-copy-id -i ~/.ssh/id_rsa.pub frankej@kisbat2.ruf.uni-freiburg.de

srun -p bosch_cpu-cascadelake -c 8 --pty bash
srun -p cpu_ivy -c 16 --pty bash
srun -p ml_gpu-rtx2080 -c 4 --gres=gpu:1 --pty bash
srun -p ml_gpu-rtx2080 -c 8 --gres=gpu:4 --pty bash
srun -p bosch_gpu-rtx2080 -c 8 --gres=gpu:4 --pty bash
srun -p gpu_tesla-P100 -c 8 --gres=gpu:4 --pty bash
srun -p alldlc_gpu-rtx2080 -c 8 --gres=gpu:1 --pty bash

rsync -av --progress --exclude *.npy --exclude *.pth frankej@kisbat2.ruf.uni-freiburg.de:/home/frankej/workspace/neuroevolution/experiments/basic_params lunarlander_32/

sbatch  -p ml_cpu-ivy -c 16 --job-name="test_j" -o meta_test.out  meta_worker.sh








