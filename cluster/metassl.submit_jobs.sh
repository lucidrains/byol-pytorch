#!/usr/bin/env bash

JOB_NAME="b001"

SESSION="ddp_resnet50_imagenet"


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

## create jobs configs
echo "create experiment configs"


if [ $USER == "frankej" ]; then
  JOBS=$(/home/frankej/miniconda3/bin/python $WORKFOLDER/cluster/cluster_create_ddp_resnet_base_training.py --project_dir $WORKFOLDER --gpus $GPUS --sess $SESSION 2>&1)
elif [ $USER == "ferreira" ]; then
  JOBS=$(/home/ferreira/.miniconda/envs/metassl/bin/python $WORKFOLDER/cluster/cluster_create_ddp_resnet_base_training.py --project_dir $WORKFOLDER --gpus $GPUS --sess $SESSION 2>&1)
fi


echo "start $JOBS jobs on $WORKFOLDER"
JOBS=`expr $JOBS - 1`

for i in `seq $START $JOBS`;
do
  LOG_FILE="/home/"$USER"/workspace/experiments/metassl/logs/""$JOB_NAME""_""$i"".out"
  ERR_FILE="/home/"$USER"/workspace/experiments/metassl/logs/""$JOB_NAME""_""$i"".err"
  echo "DO.submit: start job $i - logfile at $LOG_FILE - error file at $ERR_FILE"

  job_name="$JOB_NAME""_""$i"

  if [ $PARTITION == "bosch_gpu-rtx2080" ]; then
    sbatch -p $PARTITION -c $CPUS -t 14-00:00 --gres=gpu:$GPUS --priority=10000 --bosch --job-name=$job_name -o $LOG_FILE -e $ERR_FILE $1 $WORKFOLDER/cluster/meta_worker.sh $i $job_name $SESSION
  elif [ $PARTITION == "alldlc_gpu-rtx2080" ]; then
    sbatch -p $PARTITION -c $CPUS -t 1-00:00 --gres=gpu:$GPUS --priority=10000 --job-name=$job_name -o $LOG_FILE -e $ERR_FILE $1 $WORKFOLDER/cluster/meta_worker.sh $i $job_name $SESSION
  else
    sbatch -p $PARTITION -c $CPUS -t 6-00:00 --gres=gpu:$GPUS --priority=10000 --job-name=$job_name -o $LOG_FILE -e $ERR_FILE $1 $WORKFOLDER/cluster/meta_worker.sh $i $job_name $SESSION
  fi
  sleep 1
done
