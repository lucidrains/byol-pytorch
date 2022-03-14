#!/usr/bin/env bash
EXPT_NAME=$1
EPOCHS=$2
LR=$3
SSL_MODEL=$4

echo "EXPT_NAME $1"
echo "EPOCHS $2"
echo "LR $3"
echo "pretrained SSL MODEL $SSL_MODEL"

#PARTITION="ml_gpu-rtx2080"
PARTITION="bosch_gpu-rtx2080"
#PARTITION="gpu_tesla-P100"
#PARTITION="meta_gpu-ti"
#PARTITION="alldlc_gpu-rtx2080"
#PARTITION="mldlc_gpu-rtx2080"
#PARTITION="testdlc_gpu-rtx2080"

GPUS=8

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH


WORKFOLDER="/home/$USER/workspace/metassl"
export PYTHONPATH=$PYTHONPATH:$WORKFOLDER

if [ $USER == "frankej" ]; then
  source /home/frankej/miniconda3/etc/profile.d/conda.sh
  conda activate metassl
elif [ $USER == "ferreira" ]; then
  source /home/ferreira/.miniconda/bin/activate metassl
fi

LOG_FILE="/home/"$USER"/workspace/experiments/metassl/logs/"$EXPT_NAME".out"
ERR_FILE="/home/"$USER"/workspace/experiments/metassl/logs/"$EXPT_NAME".err"
SSL_MODEL_FILE=$EXPT$SSL_MODEL

echo "submitted job $EXPT_NAME"
echo "logfile at $LOG_FILE"
echo "error file at $ERR_FILE"

if [ $PARTITION == "bosch_gpu-rtx2080" ]; then
  sbatch -p $PARTITION --gres=gpu:$GPUS --priority=10000 --bosch --job-name=$EXPT_NAME -o $LOG_FILE -e $ERR_FILE $WORKFOLDER/cluster/train_simsiam.sh $EXPT_NAME $EPOCHS $LR $SSL_MODEL
elif [ $PARTITION == "alldlc_gpu-rtx2080" ]; then
  sbatch -p $PARTITION --gres=gpu:$GPUS --priority=10000 --job-name=$EXPT_NAME -o $LOG_FILE -e $ERR_FILE $WORKFOLDER/cluster/train_simsiam.sh $EXPT_NAME $EPOCHS $LR $SSL_MODEL
else
  sbatch -p $PARTITION --gres=gpu:$GPUS --priority=10000 --job-name=$EXPT_NAME -o $LOG_FILE -e $ERR_FILE $WORKFOLDER/cluster/train_simsiam.sh $EXPT_NAME $EPOCHS $LR $SSL_MODEL
fi



