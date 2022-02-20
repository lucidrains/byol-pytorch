#!/bin/bash
#SBATCH -p alldlc_gpu-rtx3080
#SBATCH --gres=gpu:8
#SBATCH --job-name=imagenet-alternating-learnaug-colorjitter-ftlr-100-ptlr-5e-2-bs-256-epochs-400-warmup-10
#SBATCH -o /work/dlclarge2/ferreira-metassl/metassl/experiments/logs/%x.%N.%A.%a.out
#SBATCH --array=0-10%1

TRAIN_EPOCHS=400
FINETUNING_EPOCHS=400
WARMUP_EPOCHS=10
EXPT_NAME="imagenet-alternating-learnaug-colorjitter-ftlr-100-ptlr-5e-2-bs-256-epochs-$TRAIN_EPOCHS-warmup-$WARMUP_EPOCHS"
LEARNAUG_TYPE="colorjitter"

echo "TRAIN EPOCHS $TRAIN_EPOCHS"
echo "FINETUNING EPOCHS $FINETUNING_EPOCHS"
echo "EXPT NAME $EXPT_NAME"
echo "WARMUP EPOCHS $WARMUP_EPOCHS"

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
echo "running srun with command: srun $WORKFOLDER/cluster/train_imagenet_alternating_simsiam.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS $WARMUP_EPOCHS $LEARNAUG_TYPE"

srun $WORKFOLDER/cluster/train_imagenet_alternating_simsiam.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS $WARMUP_EPOCHS $LEARNAUG_TYPE
~
