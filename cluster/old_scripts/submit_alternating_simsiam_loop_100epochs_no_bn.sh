#!/bin/bash 
TRAIN_EPOCHS=100
FINETUNING_EPOCHS=100
EXPT_NAME="alternating-advanced-stats-ftlr-100-ptlr-5e-2-bs-256-epochs-100-no-bn"
#EXPT_NAME="test"

# echo "TRAIN EPOCHS $TRAIN_EPOCHS"
# echo "FINETUNING EPOCHS $FINETUNING_EPOCHS"
# echo "EXPT NAME $EXPT_NAME"

#PARTITION="ml_gpu-rtx2080"
#PARTITION="bosch_gpu-rtx2080"
#PARTITION="gpu_tesla-P100"
#PARTITION="meta_gpu-ti"
PARTITION="alldlc_gpu-rtx2080"
#PARTITION="mldlc_gpu-rtx2080"
#PARTITION="testdlc_gpu-rtx2080"


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

WORKFOLDER="/home/$USER/workspace/metassl"
# echo $WORKFOLDER
export PYTHONPATH=$PYTHONPATH:$WORKFOLDER

source /home/ferreira/.miniconda/bin/activate metassl

# echo "submitted job $EXPT_NAME"
# echo "logfile at $LOG_FILE"
# echo "error file at $ERR_FILE"

# hostname
timeout 23h srun -p $PARTITION --exclude=dlcgpu24,dlcgpu22,dlcgpu28,dlcgpu05,dlcgpu23,dlcgpu31 --gres=gpu:8 --job-name=$EXPT_NAME -o /home/ferreira/workspace/experiments/metassl/logs/%x.%N.%A.%a.out $WORKFOLDER/cluster/train_alternating_simsiam_no_bn.sh $EXPT_NAME $TRAIN_EPOCHS $FINETUNING_EPOCHS
if [[ $? -eq 124 ]]; then
	# echo "timeout"
	# hostname
	$WORKFOLDER/cluster/submit_alternating_simsiam_loop_100epochs_no_bn.sh
fi
