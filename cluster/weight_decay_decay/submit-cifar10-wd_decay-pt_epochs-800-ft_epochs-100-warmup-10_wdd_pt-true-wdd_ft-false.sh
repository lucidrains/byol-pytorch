#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --job-name=wd_decay-cifar10-pt_epochs-800-ft_epochs-100-warmup-10-wdd_pt-true-wdd_ft-false
#SBATCH -o /work/dlclarge2/ferreira-metassl/metassl/experiments/logs/%x.%N.%A.%a.out
#SBATCH --array=0-3%1

TRAIN_EPOCHS=800
FINETUNING_EPOCHS=100
WARMUP_EPOCHS=10
EXPT_NAME="wd_decay-cifar10-pt_epochs-800-ft_epochs-100-warmup-10-wdd_pt-true-wdd_ft-false"
CONFIG="metassl/default_metassl_config_cifar10.yaml"

echo "TRAIN EPOCHS $TRAIN_EPOCHS"
echo "FINETUNING EPOCHS $FINETUNING_EPOCHS"
echo "EXPT NAME $EXPT_NAME"
echo "WARMUP EPOCHS $WARMUP_EPOCHS"
echo "CONFIG $CONFIG"

#PARTITION="ml_gpu-rtx2080"
#PARTITION="bosch_gpu-rtx2080"
#PARTITION="gpu_tesla-P100"
#PARTITION="meta_gpu-ti"
#PARTITION="alldlc_gpu-rtx2080"
#PARTITION="mldlc_gpu-rtx2080"
#PARTITION="testdlc_gpu-rtx2080"


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

source variables.sh

echo $WORKFOLDER
export PYTHONPATH=$PYTHONPATH:$WORKFOLDER

source /home/ferreira/.miniconda/bin/activate metassl

echo "submitted job $EXPT_NAME"

srun $WORKFOLDER/cluster/weight_decay_decay/cifar_train_finetune_wdd_pt_true_wdd_ft_false.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS $WARMUP_EPOCHS $CONFIG
