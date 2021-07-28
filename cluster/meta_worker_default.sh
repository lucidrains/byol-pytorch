#!/usr/bin/env bash

echo "meta worker: start job-id $1 -name $2 -session 3"
if [ $USER == "frankej" ]; then
  source /home/frankej/miniconda3/etc/profile.d/conda.sh
  conda activate metassl
elif [ $USER == "ferreira" ]; then
  source /home/ferreira/.miniconda/bin/activate metassl
fi

python $PROJECT/cluster/cluster_start_default_config.py --data_dir=$1 --expt_dir=$2 --project_dir=$3 --job_name=$4 --sess=$5

