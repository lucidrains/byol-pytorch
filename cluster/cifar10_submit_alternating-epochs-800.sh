#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH --job-name=cifar10-alternating-epochs-800
#SBATCH -o /work/dlclarge2/ferreira-metassl/metassl/experiments/logs/%x.%N.%A.%a.out
#SBATCH --array=0-10%1

EXPT_NAME="cifar10-alternating-epochs-800"
TRAIN_EPOCHS=800
FINETUNING_EPOCHS=800
WARMUP_EPOCHS=10
LEARNAUG_TYPE="default"

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

echo "WORKFOLDER $WORKFOLDER"
export PYTHONPATH=$PYTHONPATH:$WORKFOLDER

source /home/ferreira/.miniconda/bin/activate metassl

echo "submitted job $EXPT_NAME"
echo "running srun with command: srun $WORKFOLDER/cluster/train_cifar10_alternating_simsiam.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS $WARMUP_EPOCHS $LEARNAUG_TYPE"
srun $WORKFOLDER/cluster/train_cifar10_alternating_simsiam.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS $WARMUP_EPOCHS $LEARNAUG_TYPE
