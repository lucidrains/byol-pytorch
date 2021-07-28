#!/usr/bin/env bash

DATA="/home/"$USER"/workspace/data/metassl"
EXPT="/home/"$USER"/workspace/experiments/"
PROJECT="/home/"$USER"/workspace/metassl"

JOB_NAME="001"
SESSION="resnet50_supervised_imagenet"

#PARTITION="ml_gpu-rtx2080"
#PARTITION="bosch_gpu-rtx2080"
#PARTITION="gpu_tesla-P100"
#PARTITION="meta_gpu-ti"
PARTITION="alldlc_gpu-rtx2080"
#PARTITION="mldlc_gpu-rtx2080"

GPUS=8
CPUS=64

START=0


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384

WORKFOLDER="/home/$USER/workspace/metassl"

if [ $USER == "frankej" ]; then
  source /home/frankej/miniconda3/etc/profile.d/conda.sh
  conda activate metassl
elif [ $USER == "ferreira" ]; then
  source /home/ferreira/.miniconda/bin/activate metassl
fi

LOG_FILE="/home/"$USER"/workspace/experiments/metassl/logs/"$JOB_NAME".out"
ERR_FILE="/home/"$USER"/workspace/experiments/metassl/logs/"$JOB_NAME".err"
echo "DO.submit: start default config job $JOB_NAME - logfile at $LOG_FILE - error file at $ERR_FILE"

if [ $PARTITION == "bosch_gpu-rtx2080" ]; then
  sbatch -p $PARTITION -c $CPUS -t 14-00:00 --gres=gpu:$GPUS --priority=10000 --bosch --job-name=$JOB_NAME -o $LOG_FILE -e $ERR_FILE $1 $WORKFOLDER/cluster/meta_worker_default.sh $DATA $EXPT $PROJECT $JOB_NAME $SESSION
elif [ $PARTITION == "alldlc_gpu-rtx2080" ]; then
  sbatch -p $PARTITION -c $CPUS -t 1-00:00 --gres=gpu:$GPUS --priority=10000 --job-name=$JOB_NAME -o $LOG_FILE -e $ERR_FILE $1 $WORKFOLDER/cluster/meta_worker_default.sh $DATA $EXPT $PROJECT $JOB_NAME $SESSION
else
  sbatch -p $PARTITION -c $CPUS -t 6-00:00 --gres=gpu:$GPUS --priority=10000 --job-name=$JOB_NAME -o $LOG_FILE -e $ERR_FILE $1 $WORKFOLDER/cluster/meta_worker_default.sh $DATA $EXPT $PROJECT $JOB_NAME $SESSION
fi



