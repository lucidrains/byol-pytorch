#!/bin/bash
TRAIN_EPOCHS=200
FINETUNING_EPOCHS=200
EXPT_NAME="alternating-ftlr-100-ptlr-5e-2-bs-256-epochs-200"

echo "TRAIN EPOCHS $TRAIN_EPOCHS"
echo "FINETUNING EPOCHS $FINETUNING_EPOCHS"
echo "EXPT NAME $EXPT_NAME"

#PARTITION="ml_gpu-rtx2080"
#PARTITION="bosch_gpu-rtx2080"
#PARTITION="gpu_tesla-P100"
#PARTITION="meta_gpu-ti"
PARTITION="alldlc_gpu-rtx2080"
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

time=`date +"%Y-%m-%d-%H-%M-%S"`
echo $time
LOG_FILE="/home/"$USER"/workspace/experiments/metassl/logs/"$EXPT_NAME"_$time.out"
ERR_FILE="/home/"$USER"/workspace/experiments/metassl/logs/"$EXPT_NAME"_$time.err"
SSL_MODEL_FILE=$EXPT$SSL_MODEL

echo "submitted job $EXPT_NAME"
echo "logfile at $LOG_FILE"
echo "error file at $ERR_FILE"

if [ $PARTITION == "bosch_gpu-rtx2080" ]; then
  timeout 90h sbatch -p $PARTITION --gres=gpu:$GPUS --priority=10000 --bosch --job-name=$EXPT_NAME -o $LOG_FILE -e $ERR_FILE $WORKFOLDER/cluster/train_alternating_simsiam.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS
  if [[ $? -eq 124 ]]; then
    ./submit_alternating_simsiam_loop.sh
  fi
elif [ $PARTITION == "alldlc_gpu-rtx2080" ]; then
	echo srun -p $PARTITION --gres=gpu:$GPUS --priority=10000 --job-name=$EXPT_NAME -o $LOG_FILE -e $ERR_FILE $WORKFOLDER/cluster/train_alternating_simsiam.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS
	timeout 5m srun -p $PARTITION --gres=gpu:$GPUS --priority=10000 --job-name=$EXPT_NAME -o $LOG_FILE -e $ERR_FILE $WORKFOLDER/cluster/train_alternating_simsiam.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS
  	if [[ $? -eq 124 ]]; then
    		./submit_alternating_simsiam_loop.sh
  	fi
else
  timeout 23h sbatch -p $PARTITION --gres=gpu:$GPUS --priority=10000 --job-name=$EXPT_NAME -o $LOG_FILE -e $ERR_FILE $WORKFOLDER/cluster/train_alternating_simsiam.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS
  if [[ $? -eq 124 ]]; then
    ./submit_alternating_simsiam_loop.sh
  fi
fi



