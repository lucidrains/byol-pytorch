#!/usr/bin/env bash

DATA="/home/"$USER"/workspace/data/metassl"
EXPT="/home/"$USER"/workspace/experiments/"
PROJECT="/home/"$USER"/workspace/metassl"

echo "meta worker: start job-id $1 -name $2 -session 3"
if [ $USER == "frankej" ]; then
  source /home/frankej/miniconda3/etc/profile.d/conda.sh
  conda activate metassl
elif [ $USER == "ferreira" ]; then
  source /home/ferreira/.miniconda/bin/activate metassl
fi

python $PROJECT/cluster/cluster_start_job.py --job_idx=$1 --data_dir=$DATA --expt_dir=$EXPT --project_dir=$PROJECT --job_name=$2 --sess=$3

