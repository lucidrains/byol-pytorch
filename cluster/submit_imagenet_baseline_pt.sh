#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:8
#SBATCH -J MSSL_ImageNet_SimSiam
#SBATCH -t 23:59:59

source activate metassl

python -m metassl.train_simsiam --config "metassl/default_metassl_config_imagenet.yaml" \
				--use_fixed_args \
				--data.dataset_percentage_usage 100 \
				--train.epochs 100 \
				--expt.warmup_epochs 0 \
				--expt.seed 0 \
				--expt.save_model_frequency 5 \
				--expt.is_non_grad_based \
				--expt.expt_name $EXPERIMENT_NAME

