#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH --job-name=alternating-ftlr-1e-5-ptlr-1e-5-bs-256-epochs-100-no-bn-small-lr
#SBATCH -o /home/ferreira/workspace/experiments/metassl/logs/%x.%N.%A.%a.out
#SBATCH --array=0-8%1

TRAIN_EPOCHS=100
FINETUNING_EPOCHS=100
TRAIN_LR=0.00001
FINETUNING_LR=0.00001

EXPT_NAME="alternating-ftlr-1e-5-ptlr-1e-5-bs-256-epochs-100-no-bn-small-lr"

echo "TRAIN EPOCHS $TRAIN_EPOCHS"
echo "FINETUNING EPOCHS $FINETUNING_EPOCHS"
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

#export NCCL_DEBUG=INFO

WORKFOLDER="/home/$USER/workspace/metassl"
echo $WORKFOLDER
export PYTHONPATH=$PYTHONPATH:$WORKFOLDER

source /home/ferreira/.miniconda/bin/activate metassl

echo "submitted job $EXPT_NAME"
echo "logfile at $LOG_FILE"
echo "error file at $ERR_FILE"

srun $WORKFOLDER/cluster/train_alternating_simsiam_no_bn_small_lr.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS $TRAIN_LR $FINETUNING_LR
