#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH --job-name=cifar10-alternating-parameterized-aug-nn-epochs-400-warmup-10
#SBATCH -o /work/dlclarge2/ferreira-metassl/experiments/logs/%x.%N.%A.%a.out
#SBATCH --array=0-10%1

TRAIN_EPOCHS=400
WARMUP_EPOCHS=10
EXPT_NAME="cifar10-alternating-parameterized-aug-nn-epochs-$TRAIN_EPOCHS-warmup-$WARMUP_EPOCHS"

echo "EXPT NAME $EXPT_NAME"

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

srun $WORKFOLDER/cluster/train_cifar10_alternating_simsiam_warmup_parameterized_aug_default_config_epochs_warmup.sh $EXPT_NAME $TRAIN_EPOCHS $WARMUP_EPOCHS
