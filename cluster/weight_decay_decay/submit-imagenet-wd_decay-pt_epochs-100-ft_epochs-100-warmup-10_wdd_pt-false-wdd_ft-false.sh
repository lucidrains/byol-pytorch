#!/bin/bash
#SBATCH -p alldlc_gpu-rtx3080
#SBATCH --gres=gpu:8
#SBATCH --job-name=wd_decay-imagenet-pt_epochs-100-ft_epochs-100-warmup-10-wdd_pt-false-wdd_ft-false
#SBATCH -o /work/dlclarge2/ferreira-metassl/metassl/experiments/logs/%x.%N.%A.%a.out
#SBATCH --array=0-10%1
#SBATCH --exclude=dlcgpu19

TRAIN_EPOCHS=100
FINETUNING_EPOCHS=100
WARMUP_EPOCHS=10
EXPT_NAME="wd_decay-imagenet-pt_epochs-100-ft_epochs-100-warmup-10-wdd_pt-false-wdd_ft-false"
CONFIG="metassl/default_metassl_config_imagenet.yaml"

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
echo "command used: srun $WORKFOLDER/cluster/weight_decay_decay/imagenet_train_finetune_wdd_pt_false_wdd_ft_false.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS $WARMUP_EPOCHS $CONFIG"
srun $WORKFOLDER/cluster/weight_decay_decay/imagenet_train_finetune_wdd_pt_false_wdd_ft_false.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS $WARMUP_EPOCHS $CONFIG
