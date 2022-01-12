#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-dsengupt
#SBATCH --gres=gpu:1
#SBATCH -o /work/dlclarge1/dsengupt-lth_ws/slurm_logs/simsiam_train.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-lth_ws/slurm_logs/simsiam_train.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J SimSiam_Cifar10
#SBATCH -t 19:59:00
#SBATCH --mail-type=BEGIN,END,FAIL

cd $(ws_find lth_ws)
#python3 -m venv lth_env
source lth_env/bin/activate
pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
cd MetaSSL/metassl/
echo "Pretrain Simsiam with CIFAR10 and Knn Run 2"
python3 -m metassl.train_simsiam --expt_name run2_simsiam_cifar10 --epochs 500 --expt_mode CIFAR10 --workers 8 --seed 125 --run_knn_val
# python -m metassl.train_linear_classifier_simsiam --expt_name "test" --ssl_model_checkpoint_path "/work/dlclarge1/dsengupt-lth_ws/MetaSSL/metassl/experiments/sanity/checkpoint_0000.pth.tar" --target_model_checkpoint_path "" --epochs 2 --expt_mode CIFAR10 --download_data --workers 8
deactivate
